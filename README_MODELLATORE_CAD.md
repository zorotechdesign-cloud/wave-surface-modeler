# Modellatore CAD con rilievo locale

Questa è una seconda applicazione: il generatore originale `onde_generator.py`
rimane disponibile e non viene sostituito.

## Funzioni disponibili

- apertura diretta di file STEP/STP e IGES/IGS;
- riconoscimento separato delle facce CAD;
- selezione visuale di una faccia con il mouse;
- selezione certa con i pulsanti **Faccia precedente/successiva** quando il
  clic 3D è ambiguo;
- scelta precisa del centro e del raggio della zona da lavorare;
- alternativa per lavorare tutta la faccia;
- anteprima del rilievo con evidenziazione della zona modificata;
- regolazione di risoluzione, densità, complessità e altezza;
- esportazione del solido lavorato come STL binario.

La lavorazione usa coordinate parametriche, normali della vera faccia CAD e un
campo tridimensionale: la fantasia non viene più trattata come un foglio piano
avvolto sul pezzo. Le seam parametriche di cilindri e coni vengono riconosciute
e attraversate senza tagli. Solo i bordi fisici ricevono una dissolvenza dolce,
così la faccia rimane collegata alle facce adiacenti.

## Installazione

Da PowerShell, nella cartella del progetto:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

## Avvio

Fare doppio clic su `avvia_modellatore_cad.bat`, oppure eseguire:

```powershell
.\.venv\Scripts\python.exe modellatore_cad.py
```

## Procedura

1. Trascinare il file STEP/IGES direttamente sulla finestra, oppure premere
   **Sfoglia cartelle…**. Per usare gli appunti, copiare il file o il suo
   percorso da Esplora file e premere **Incolla e carica**. È anche possibile
   scrivere il percorso nel campo bianco e premere **Invio** o **Carica**. Sono
   accettati i percorsi tra virgolette prodotti da **Copia come percorso**.
2. Impostare la risoluzione della sola anteprima prima dell'importazione. Un
   valore più piccolo produce più dettaglio ma richiede più memoria e tempo.
3. Cliccare la faccia CAD da lavorare oppure scorrere le facce con i pulsanti
   **Faccia precedente/successiva**. Il punto rosso indica il centro della zona.
4. Scegliere **Zona circolare** e il raggio, oppure **Intera faccia**.
5. Regolare densità, complessità e altezza del rilievo.
6. Premere **Applica / aggiorna anteprima**.
7. Controllare la zona arancione/rossa ed esportare lo STL lavorato. Se non è
   stata ancora generata l'anteprima, il pulsante di esportazione la applica
   automaticamente. Lo STL viene salvato nella stessa cartella del file CAD,
   con il suffisso `_lavorato` (e un orario se il nome esiste già).

Durante l'esportazione il programma riapre il B-Rep STEP/IGES originale e crea
una triangolazione dedicata da 0,18–0,45 mm. Non riutilizza la mesh grossolana
della preview. Su un vaso di circa 100 mm il processo può richiedere 30–60
secondi e produrre diverse centinaia di migliaia di triangoli.

## Formato di uscita

STEP e IGES vengono letti conservando l'identità B-Rep delle facce durante la
selezione. La lavorazione libera non è una superficie analitica STEP: il formato
di uscita attuale è quindi STL ad alta risoluzione. Il file è utilizzabile nei
comuni slicer e nei software CAD che importano mesh.
