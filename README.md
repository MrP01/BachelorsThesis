# Project: Secure Classification as a Service
**Student**:         Peter Julius Waldert  
**Study programme**: Information and Computer Engineering and Physics  
**Advisor**:         Daniel Kales, Roman Walch  
**Project goals**:   Classifying MNIST images on the server using homomorphic encryption.  
**Project status**:  Started on 05.11.2020

## Documentation
Give a concise description of what your tool/code/design/work achieves.

If you did not include your actual implementation in the repository because you
contributed to an external project or used your own repo, document where to find it - 
branch name, repository URL, project's name, advisor's name or similar).

If you included your code, clearly document how to compile, set up, run, use your project.
(Preferably use the ["Markdown" (MD) markup language](https://help.github.com/articles/markdown-basics/))

After your project is finished and before the repository gets archived, make sure the
`master` branch indicates where to find all your relevant resources:

- Your thesis
- Your source code contributions
- Your presentations and posters
- Any other interesting material (websites, papers, ...)

## Status
Just started out

TODOs:
- Zuerst ein einfaches Neural Network ohne homomorphic encryption trainieren (entweder in python oder C++). 
    Wie wir bereits am Donnerstag besprochen haben, müssen die Aktivierungsfunktionen (Relu, Sigmoid, etc.)
    für HE approximiert werden. Experimentiere dafür auch schon in der Plain Implementierung, welche Auswirkungen
    das auf die Accuracy des Netzwerks hat.
- Exportieren des Neural Networks (Anzahl Neuronen und Layer, die trainierten Weights, ...)
    damit dieses für das HE Netzwerk verwendet werden kann.
- Implementierung des Netzwerkes in SEAL (C++)
- Für den Web-basierten Demonstrator kannst du dann deine Implementierung nach Node-SEAL,
    einen javascript wrapper für die SEAL library, portieren. Der Demonstrator soll folgende Funktionen haben:
    Ein Client sendet ein verschlüsseltes Bild zum Server, der Server benutzt deine Implementierung
    um eine verschlüsselte Klassifizierung zu erzeugen, welche dann vom Client entschlüsselt werden kann.
    Am besten eine einfache Website zur Nutzung für den Client entwickeln.
- Für ein einfaches Deployment soll am Ende alles in einem Docker Container laufen.
- Wenn noch Zeit ist, können wir das Thema auch noch auf den smart-meetering use case von dir ausweiten,
    für uns steht in erster Linie jedoch der Demonstrator im Vordergrund.
- Am Ende musst du dann noch deine Arbeit in Form von einer schriftlichen Thesis niederschreiben
    und deine Ergebnisse präsentieren. 
