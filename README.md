Data Challenges - Object Detection / Multilabel Classification
--------------------------------------------------------------
Das ist unser Multilabel-Klassifikationsansatz, welcher für die Weiterverwendung als Backbone in Object Detection Algorithmen dienen kann. Der folgende Text beschreibt die Benutzung der im Projekt enthaltenen
Skripte, um auf die Ergebnisse zu kommen, welche wir in unserer Präsentation generiert haben. 

Zuerst muss der zu bearbeitende Datensatz lokal für das Modell eingeführt werden. Dies sollte im Unterordner /datasets/CN_dataset_obj_detection_04_23/ geschehen, 
ansonsten muss dies in jedem Python Skript korrekt umgeändert werden anhand der data_dir Variable zu Beginn des Skriptes. Damit die Daten nun von den verwendeten Modellen benutzt werden können,
muss zuerst data.py und danach dataloader.py ausgeführt werden. Für das in der Präsentation erwähnte Begrenzen der Daten auf Labels mit einer Mindestanzahl Daten sind die datamaker-Skripte zu benutzen,
wobei die Nummer im Namen die genaue Mindestanzahl bezeichnet.

Um die angesprochenen Modelle aus der Präsentation auf den verarbeiteten Datensatz zu benutzen, muss das Skript mit dem Namen des Modells ausgeführt werden. Dieses beinhaltet das Training samt Validierung
und am Ende das Testen des jeweiligen Modells. Nach Ausführung eines Modellskriptes werden die trainierten Modelle in einem Ordner namens "models" eingespeichert, einerseits über pytorch pickle und andererseits
über die huggingface save_pretrained Funktion. Sollte die oben angesprochenen labelbegrenzten Datensätze benutzt werden für die Modelle, ist dies nur für die ViT-Hybrid Modelle möglich, da die Auslesung
dieser Datensätze nur für diese Modelle konkret umgesetzt wurde. Dafür müssen die vithybrid-Skripte ausgeführt werden, welche mit Nummern enden. Diese Nummern implizieren wie vorhin für die datamaker-Skripte
die Mindestanzahl Labels.

Um die angezeigten Heatmaps zum Vergleich der Label Prediction Agreement/Confusion Matrix und der Co-occurence Matrix werden die eval-Skripte mit Nummerierungen getrennt durch "_" benutzt, wobei die Nummer die 
Begrenzung der Daten, wie zuvor angesprochen, auf eine Mindestanzahl bezeichnet. Für die Visualisierung der Anzahl Bilder pro Label und die Verteilung der Anzahl Labels pro Bild wird das visualise-Skript verwendet.
