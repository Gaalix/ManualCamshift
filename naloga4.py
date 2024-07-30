import cv2 as cv
import numpy as np

def camshift(slika, sablone, lokacije_oken, iteracije, napaka):
    '''Implementacija CamShift algoritma.'''
    hsv = cv.cvtColor(slika, cv.COLOR_BGR2HSV)
    hist_h, hist_hs = sablone
    
    # cv.calcBackProjet izračuna verjetnostno porazdelitev
    #dst = cv.calcBackProject([hsv], [0], hist_h, [0, 180], 1)
    #dst = cv.calcBackProject([hsv], [0, 1], hist_hs, [0, 180, 0, 256], 1)

    for i in range(len(lokacije_oken)):
        lokacija_okna = lokacije_oken[i]
        lokacija_okna = mean_shift(lokacija_okna, dst, iteracije, napaka)
        lokacije_oken[i] = lokacija_okna

    return lokacije_oken
    
def mean_shift(lokacija_okna, dst, iteracije, napaka, scale=0.5):
    '''Implementacija mean shift algoritma.'''
    x, y, w, h = lokacija_okna
    min_window_size = 40
    for _ in range(iteracije):
        # Zagotovimo, da okno ne gre izven slike
        x = min(max(x, 0), dst.shape[1] - w)
        y = min(max(y, 0), dst.shape[0] - h)

        # Seštevek pikslov v roi
        m00 = np.sum(dst[y:y+h, x:x+w])
        if m00 == 0:
            break

        # Težišča za izračun centroida
        #np.dot matrično množenje
        #np.arange ustvari seznam od 0 do w
        m10 = np.sum(np.dot(np.arange(w), dst[y:y+h, x:x+w].sum(axis=0)))
        m01 = np.sum(np.dot(np.arange(h), dst[y:y+h, x:x+w].sum(axis=1)))

        # Centroid x in y
        cx = m10 / m00
        cy = m01 / m00
        
        # Premik
        # np.round zaokroži na celo število
        dx = np.round(scale * (cx - w / 2)).astype(int)
        dy = np.round(scale * (cy - h / 2)).astype(int)
        
        # Sprememba velikosti okna po tem ko pridemo do konvergence
        if np.abs(dx) < napaka and np.abs(dy) < napaka:
            s = max(np.sqrt(m00/4), min_window_size)
            w = h = int(1.2 * s)
            break
        
        # Posodobitev lokacije okna
        x += dx
        y += dy
    
    return x, y, w, h

def zaznaj_gibanje(kamera, objekti=2):
    pass

def izracunaj_znacilnice(lokacije_oken, slika):
    '''Implementacija izračuna značilnic za sledenje.'''
    hsv = cv.cvtColor(slika, cv.COLOR_BGR2HSV)
    mask = np.zeros(slika.shape[:2], np.uint8)
    
    for x, y, w, h in lokacije_oken:
        mask[y:y+h, x:x+w] = 255
    
    # cv.bitwise_and ustvari ROI tako, da izračuna logični AND med masko in sliko
    roi = cv.bitwise_and(hsv, hsv, mask=mask)
    
    # Izračun histograma
    #roi.shape[0] vrne število vrstic
    hist_h = np.zeros((180, 1))
    hist_hs = np.zeros((180, 256))
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            if mask[i, j] != 0: 
                # roi[i, j, 0] je vrednost H komponente
                # roi[i, j, 1] je vrednost S komponente
                hist_h[roi[i, j, 0]] += 1
                hist_hs[roi[i, j, 0], roi[i, j, 1]] += 1
    
    # Normalizacija histograma
    hist_h = 255 * (hist_h - np.min(hist_h)) / (np.max(hist_h) - np.min(hist_h))
    hist_hs = 255 * (hist_hs - np.min(hist_hs)) / (np.max(hist_hs) - np.min(hist_hs))

    return hist_h, hist_hs

if __name__ == "__main__":
    #Naloži video
    nacin_zaznavanja = "rocno" # ali "rocno" ali "avtomatsko"
    cap = cv.VideoCapture('video.mp4')
    #Preveri, če je video uspešno naložen
    if not cap.isOpened():
        print("Napaka: Video ne obstaja ali ni bil uspešno naložen.")
        exit(1)

    #Nastavitve meanshift algoritma
    iteracije = 10
    napaka = 1
    
    lokacije_oken = list()
    #Začetna točka sledenja ročno
    #(x, y, w, h)
    if nacin_zaznavanja == "rocno":
        # (618, 14, 180, 190))
        lokacije_oken.append((618, 14, 172, 190))
        # (288, 392, 80, 60))
        lokacije_oken.append((288, 392, 76, 62))
    else:
        lokacije_oken = zaznaj_gibanje(cap, objekti=2)
    
    #Izračun značilnic za sledenje
    uspel, prva_slika = cap.read()
    sablone = izracunaj_znacilnice(lokacije_oken,prva_slika)

    
    #Začetek sledenja
    while True:
        uspel, slika = cap.read()
        if not uspel:
            break        
        
        lokacije_novih_oken = camshift(slika, sablone,lokacije_oken, iteracije, napaka)
        lokacije_oken = lokacije_novih_oken
        #Nariši okno
        for okno in lokacije_novih_oken:
            x,y,w,h = okno
            cv.rectangle(slika, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.imshow('Rezultat', slika)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
