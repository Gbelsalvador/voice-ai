# Entraîner un STT et un TTS depuis zéro — baseline PyTorch

Ce dépôt est un **socle pédagogique reproductible** pour construire deux modèles vocaux sans poids pré-entraînés :

- **STT** : audio → texte, réseau CNN + BiGRU + CTC, caractère par caractère.
- **TTS** : texte → spectrogramme Mel, encodeur BiLSTM + décodeur GRU avec attention additive. Griffin-Lim reconstruit ensuite une forme d'onde à partir du Mel.

Tous les paramètres des deux réseaux sont initialisés aléatoirement. Il n'y a ni Whisper, ni wav2vec 2.0, ni Tacotron pré-entraîné, ni poids téléchargés.

> Limite importante : le TTS apprend ici le modèle acoustique texte→Mel, mais **Griffin-Lim n'est pas un vocoder neuronal entraîné**. C'est une méthode classique d'inversion du spectrogramme. La qualité audio sera donc une baseline de recherche, souvent métallique/robotique. Pour un TTS plus naturel, le prolongement scientifique consiste à entraîner un vocoder (par exemple une architecture de type HiFi-GAN) depuis zéro sur les mêmes enregistrements. Ce dépôt ne prétend pas inclure cet entraînement de vocoder.

## 1. Environnement

Python 3.10 ou 3.11 est conseillé pour commencer. Sous Windows, Linux ou WSL :

```bash
python -m venv .venv
# Windows PowerShell : .venv\Scripts\Activate.ps1
# Linux / WSL : source .venv/bin/activate
python -m pip install --upgrade pip
```

Installer PyTorch selon la plateforme et la version CUDA de la machine depuis le sélecteur officiel : https://pytorch.org/get-started/locally/ . Pour un essai CPU :

```bash
pip install -r requirements.txt
```

Si librosa ne lit pas certains MP3, installe FFmpeg sur le système ou convertis les audios en WAV PCM mono avant l'entraînement.

Vérifier l'environnement :

```bash
python -c "import torch, librosa; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| librosa', librosa.__version__)"
```

## 2. Préparer les données

Chaque jeu de données est décrit par un CSV UTF-8 avec deux colonnes obligatoires :

```csv
audio,text
/path/vers/clip_001.wav,bonjour tout le monde
/path/vers/clip_002.wav,comment allez-vous aujourd'hui
```

Les chemins relatifs sont interprétés par rapport au dossier qui contient le CSV.

### Pour le STT

Common Voice est un point de départ pratique pour la reconnaissance vocale. Télécharge le corpus de la langue choisie, puis convertis `validated.tsv` :

```bash
python scripts/make_commonvoice_manifest.py --root "/chemin/vers/corpus/fr" --tsv validated.tsv --output manifests/stt_fr.csv
python scripts/check_manifest.py manifests/stt_fr.csv
```

Pour exploiter les splits officiels Common Voice (recommandé), crée plutôt les trois manifestes :

```bash
python scripts/make_commonvoice_splits.py --root "data/tshiluba" --output-dir manifests/tshiluba
python scripts/check_manifest.py manifests/tshiluba/train.csv
python scripts/check_manifest.py manifests/tshiluba/dev.csv
python src/train_stt.py --manifest manifests/tshiluba/train.csv --val-manifest manifests/tshiluba/dev.csv --out checkpoints/stt_tshiluba
python src/evaluate_stt.py --checkpoint checkpoints/stt_tshiluba/best.pt --manifest manifests/tshiluba/test.csv
```

`--limit` sur le script de conversion et `--limit-train` / `--limit-val` sur l'entraînement permettent de faire un essai court avant d'utiliser tout le corpus. Les caractéristiques audio sont conservées dans `data/cache/stt` et réutilisées aux lancements suivants.

Le script lit `clips/` et les colonnes `path` et `sentence`. Il peut filtrer un locuteur avec `--speaker IDENTIFIANT`.

### Pour le TTS

Le premier modèle de ce dépôt est **mono-locuteur** : il ne possède pas d'identifiant de voix ni d'embedding de locuteur. Il faut donc des enregistrements propres, cohérents, lus par une seule personne, avec leur transcription exacte. Une structure LJSpeech (`metadata.csv` et `wavs/`) peut être convertie ainsi :

```bash
python scripts/make_ljspeech_manifest.py --root "/chemin/vers/ton_corpus_tts" --output manifests/tts.csv
python scripts/check_manifest.py manifests/tts.csv
```

Pour un projet en lingala ou en swahili, tu peux constituer ton propre corpus : une phrase par fichier WAV, même locuteur, micro et distance stables, pièce calme, transcription vérifiée. Le format CSV reste le même. Évite de mélanger plusieurs langues dans un petit corpus sans définir une expérience multilingue explicite.

**Ne confonds pas corpus STT et corpus TTS :** un corpus STT peut contenir de nombreux locuteurs. Pour ce TTS baseline, utilise un seul locuteur. Le texte doit correspondre exactement à ce qui est prononcé.

### Bonnes pratiques du corpus

- Conserver les enregistrements originaux et noter les consentements/licences.
- Écarter les clips vides, saturés, avec musique forte ou transcription incorrecte.
- Pour STT, les voix et conditions variées sont utiles, mais il faut garder un test représentatif.
- Pour TTS, enregistrer des phrases variées, avec ponctuation et toutes les lettres/sons visés.
- Séparer train, validation et test avant de tirer des conclusions. Le script d'entraînement réserve 5 % pour validation ; garde aussi un **jeu test indépendant** pour le rapport final.

## 3. Entraîner le STT depuis zéro

Depuis la racine du projet :

```bash
python src/train_stt.py \
  --manifest manifests/stt_fr.csv \
  --out checkpoints/stt_fr \
  --epochs 40 \
  --batch-size 8 \
  --lr 0.001
```

Sous PowerShell, mets la commande sur une seule ligne ou utilise l'accent grave comme caractère de continuation.

Le code :
1. lit les paires audio/transcription ;
2. extrait des log-Mel de 80 bandes à 16 kHz ;
3. construit un vocabulaire de caractères à partir du train ;
4. entraîne le CNN-BiGRU avec la loss CTC ;
5. écrit `best.pt`, `vocab.json` et `history.json` dans le dossier de sortie.

Reconnaître un fichier :

```bash
python src/infer_stt.py --checkpoint checkpoints/stt_fr/best.pt --audio data/test.wav
```

Évaluer sur un CSV de test séparé :

```bash
python src/evaluate_stt.py --checkpoint checkpoints/stt_fr/best.pt --manifest manifests/stt_test.csv
```

Métriques : **WER** (Word Error Rate) et **CER** (Character Error Rate), en pourcentage. Plus bas est meilleur. N'évalue jamais sur les audios ayant servi à entraîner le modèle.

## 4. Entraîner le TTS depuis zéro

Le TTS attend un corpus d'une seule voix, avec les fichiers audio accessibles dans le manifest :

```bash
python src/train_tts.py \
  --manifest manifests/tts_fr.csv \
  --out checkpoints/tts_fr \
  --epochs 100 \
  --batch-size 4 \
  --lr 0.0002
```

Le code :
1. convertit les textes en caractères ;
2. rééchantillonne les audios à 22,05 kHz ;
3. extrait des log-Mel à 80 bandes ;
4. entraîne un encodeur BiLSTM et un décodeur GRU avec attention ;
5. prédit les trames Mel et un signal d'arrêt ;
6. sauvegarde le meilleur checkpoint.

Générer une phrase :

```bash
python src/infer_tts.py --checkpoint checkpoints/tts_fr/best.pt --text "bonjour, comment allez-vous ?" --output generated.wav
```

Le résultat se trouve dans `generated.wav`. Au début, le modèle peut prononcer incorrectement, répéter des sons ou s'arrêter trop tôt. C'est normal avec un petit corpus et une architecture pédagogique. Il faut observer la loss de validation, écouter plusieurs phrases jamais vues et augmenter la taille/qualité du corpus avant de conclure.

## 5. Conseils d'entraînement et ressources matérielles

- Commence avec un petit sous-ensemble pour vérifier que le pipeline fonctionne, puis augmente progressivement.
- Sur CPU, réduis `--batch-size` (par exemple 2 pour le TTS, 2 ou 4 pour le STT) et le nombre d'exemples. L'entraînement complet peut être long.
- En cas de mémoire GPU insuffisante, réduis le batch size et la durée maximale des clips. Le TTS autoregressif est particulièrement lent car il prédit une trame à la fois.
- Un bon test de débogage consiste à sur-apprendre 20 à 50 phrases : la loss train doit nettement diminuer et le modèle doit réussir à reproduire ces phrases. Cela ne mesure pas sa généralisation.
- Garde les mêmes paramètres Mel entre entraînement et inférence. Ils sont définis dans le code et stockés dans le checkpoint.

## 6. Protocole expérimental conseillé pour un mémoire

1. **Question de recherche** : par exemple, dans quelle mesure un modèle STT CTC de petite taille peut-il reconnaître le lingala dans des conditions propres puis bruitées ?
2. **Corpus** : documenter nombre d'heures, nombre de locuteurs, durée moyenne, dialecte/variété, conditions d'enregistrement, licence et consentement.
3. **Découpage** : séparer par locuteur lorsque c'est possible, pas uniquement par phrase, afin d'éviter qu'une même voix apparaisse dans train et test.
4. **Baselines** : comparer ton modèle entraîné depuis zéro avec une baseline reconnue (par exemple un modèle pré-entraîné), mais garder les résultats clairement séparés : la baseline n'est pas ton modèle entraîné depuis zéro.
5. **Mesures STT** : WER, CER, latence, taille du modèle, performance selon bruit/accent/locuteur.
6. **Mesures TTS** : écoute aveugle par plusieurs évaluateurs, intelligibilité, naturel (MOS si protocole réalisable), erreurs de prononciation, latence et taille. Pour une analyse acoustique, ajouter des mesures adaptées comme MCD avec prudence ; aucune métrique unique ne remplace l'écoute humaine.
7. **Ablations** : quantité de données, normalisation, bruit, taille du modèle et taille du vocabulaire.

## 7. Arborescence

```text
voice_ai_from_scratch/
├── src/
│   ├── common.py
│   ├── stt_model.py
│   ├── train_stt.py
│   ├── infer_stt.py
│   ├── evaluate_stt.py
│   ├── tts_model.py
│   ├── train_tts.py
│   └── infer_tts.py
├── scripts/
│   ├── make_commonvoice_manifest.py
│   ├── make_ljspeech_manifest.py
│   └── check_manifest.py
├── manifests/
├── data/
├── checkpoints/
└── requirements.txt
```

## 8. Limites scientifiques à annoncer honnêtement

Ce code est volontairement une **baseline éducative**, pas un système commercial ni un état de l'art. Le STT CTC n'est pas un Transformer/Conformer. Le TTS utilise une attention simple, n'a pas de prosodie explicite ni de conditionnement multi-locuteur, et utilise Griffin-Lim au lieu d'un vocoder neuronal entraîné. Les résultats seront fortement déterminés par la quantité, la qualité et la cohérence des données.

Pour un PFE sérieux, une contribution défendable serait de construire et documenter un corpus validé pour une langue peu dotée, d'étudier le comportement du STT selon les conditions acoustiques, puis d'évaluer le TTS mono-locuteur. Il vaut mieux obtenir un STT bien évalué et un TTS expérimental clairement limité que présenter deux démos sans protocole scientifique.
