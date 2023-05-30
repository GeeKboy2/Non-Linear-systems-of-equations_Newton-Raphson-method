## Organisation du dépôt

Le code du projet se trouve dans le répertoire **src**. Pour chaque partie, il y a un code correspondant nommé au numéro de sa partie.
Par exemple, le code de la partie 1 se trouve dans le fichier `part1.py`.
On y trouve aussi les différents fichiers de tests qui correspondent aux différentes parties.
Par exemple, les tests pour la partie 1 sont dans le fichier `test1.py`. 

Dans le répertoire **sections**, on peut trouver les différents fichiers latex correspondant aux différentes parties incluses dans le fichier `rapport.tex`.

## Makefile

Le **Makefile** dispose de plusieurs cibles, et permet en exécutant la commande `make test` de lancer des tests sur toutes les parties. 
La commande `make` permet de générer le rapport, et pour obtenir des détails sur la compilation du rapport, il faut lancer la commande `make verbose`.