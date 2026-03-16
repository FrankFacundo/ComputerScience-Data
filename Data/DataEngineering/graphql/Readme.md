Pour maîtriser GraphQL, voici la liste des notions essentielles à comprendre en français :

### 1. Le Schéma (Schema)

C'est la pièce maîtresse de GraphQL. Le schéma agit comme un contrat strict entre le client (front-end) et le serveur (back-end). Il définit exactement quelles données peuvent être demandées ou modifiées, ainsi que leurs types. Le client sait donc à l'avance ce qu'il peut faire sans avoir à deviner.

### 2. Les Types (Types & Scalars)

GraphQL est fortement typé. Vous définissez des "Types" pour représenter vos objets (par exemple, un type `Utilisateur` qui possède un `nom` et un `age`).
Il existe des types de base (les scalaires) : `String`, `Int`, `Float`, `Boolean`, et `ID`. Vous pouvez aussi créer vos propres types complexes et définir si un champ est obligatoire (avec un `!`).

### 3. Les Requêtes (Queries)

C'est l'équivalent d'une requête `GET` dans une API REST classique. Une *Query* permet au client de demander uniquement les données dont il a besoin, rien de plus, rien de moins. Cela évite l'over-fetching (récupérer trop de données) et l'under-fetching (ne pas en avoir assez et devoir faire plusieurs appels).

### 4. Les Mutations

C'est l'équivalent des requêtes `POST`, `PUT`, `PATCH` ou `DELETE` en REST. Une *Mutation* est utilisée pour modifier des données côté serveur (créer un utilisateur, mettre à jour un article, supprimer un commentaire) et elle retourne la donnée modifiée dans la foulée.

### 5. Les Abonnements (Subscriptions)

Les *Subscriptions* permettent de maintenir une connexion en temps réel avec le serveur (généralement via des WebSockets). Chaque fois qu'un événement précis se produit côté serveur (par exemple, un nouveau message dans un chat), le serveur pousse instantanément la nouvelle donnée au client.

### 6. Les Variables

Au lieu d'écrire des valeurs "en dur" dans vos requêtes ou vos mutations (comme l'ID d'un utilisateur), vous utilisez des variables. Cela permet de réutiliser la même requête dynamique tout en passant un dictionnaire JSON contenant les paramètres.

### 7. Les Fragments

Si vous avez besoin des mêmes champs à plusieurs endroits dans vos requêtes, vous pouvez les regrouper dans un *Fragment*. C'est un bloc de requête réutilisable qui permet de garder votre code propre (le principe DRY : *Don't Repeat Yourself*).

---

### 📺 Recommandations de tutoriels YouTube en français

Pour voir tout cela en action, voici quelques excellentes vidéos en français que je vous recommande, trouvées sur YouTube :

1. **"C'est quoi une API GraphQL ?"** par *Kodaps - apprendre à coder*
Une excellente introduction théorique pour bien comprendre pourquoi GraphQL a été créé par Facebook pour pallier les limites des API REST, et comment ça fonctionne de manière générale.
👉 [https://www.youtube.com/watch?v=GC8k7mAP464](https://www.youtube.com/watch?v=GC8k7mAP464)
2. **"📣 C'EST QUOI LE GRAPHQL ? (TUTORIEL)"** par *Algoconcept*
Une vidéo un peu plus longue (20 minutes) qui plonge plus en détail dans les concepts et l'architecture de GraphQL pour les développeurs.
👉 [https://www.youtube.com/watch?v=-OIL79OlWIc](https://www.youtube.com/watch?v=-OIL79OlWIc)

Ces vidéos vous donneront une excellente base visuelle et pratique pour relier toutes ces notions, notamment la façon dont les requêtes tapent dans le schéma, qui fait lui-même appel à vos fameux *resolvers* !