+++
title = 'Finding Meaningful Directions in an Embedding Space'
date = 2025-07-06T01:05:29+01:00
draft = false
+++


Word embeddings are the backbone of many NLP applications, but they often lack interpretability. We all know the famous `"king" - "man" + "woman" = "queen"` analogy, but how can we uncover similar relationships in a more systematic, unsupervised way? 

This notebook explores how to find and interpret intrinsic directions in the word embedding space using clustering techniques. 

You can find all the directions I found and their interpretations [in the results json](https://github.com/PhilSad/latent-directions/blob/main/all_relevant_clusters_with_meaning.json).

# High-level overview

## The embedding model
I need a model that can provide high-quality word embeddings. Since I'm only working at the word level, I don't need a model like BERT that provides contextual embeddings. Instead, I can use a pre-trained word embedding model that captures semantic relationships between words.

I used the [Google News Word2Vec model](https://code.google.com/archive/p/word2vec/), which is a pre-trained word embedding model trained on a large corpus of news articles. The embedding dimension is 300.

## The data
To keep things simple, I only worked with nouns, using [this list](https://raw.githubusercontent.com/lukecheng1998/20-Questions/refs/heads/master/nouns.txt) of 6800 common English nouns and filtered out any words that were not in the embedding model.

## The method
### The idea
The idea is to find groups of words that have a similar direction in the embedding space. A group would look like this:
- "man" > "woman"
- "dad" > "mom"
- "brother" > "sister"

Then we can find the direction of the group by calculating the difference between the embeddings of the words and averaging them. This gives us a vector that represents the direction of the group.

We could then use this vector to find other words: 
- `"king"  + vector = "queen"`
- `"uncle" + vector = "aunt"`

### What didn't work
I initially tried to calculate the pairwise difference between all words embeddings and then cluster the resulting vectors. However, this approach was too memory-intensive, requiring around 2TB of RAM to cluster the 6800*6800 pairwise directions.

Maybe I could have searched for a more memory-efficient clustering algorithm, but I had another idea that worked much better.

Also I tried BERT embeddings but the `"king" - "man" + "woman" = "queen"` case didn't work at all. BERT embeddings might be on a higher level of abstraction rendering simple arithmetic operations ineffective. The embeddings are context-dependent, and the relationships between words are not as straightforward as in Word2Vec.

### What did work
Instead, I first clustered the words into similar groups, and then performed another clustering  on the pairwise differences within each group. This approach was much more memory-efficient and allowed me to find meaningful directions.

After some filtering, I got 90 clusters. I can then use a LLM to interpret the clusters and find their meaning. Some of the clusters are quite interesting! You can find the results at the end of this notebook.

# Let's get started!

## Load the data and the model

```python
import gensim.downloader as api
import numpy as np
import hdbscan
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

wv = api.load('word2vec-google-news-300')
```


```python
# do dbscan clustering with wv similarity distance
with open("./nounlist.txt", "r") as f:
    all_words = f.read().splitlines()
all_words = [word for word in all_words if word in wv]
```

## Reduce the dimensionality of the embeddings (cf: the curse of dimensionality)

```python
from sklearn.decomposition import PCA, IncrementalPCA
X = np.array([wv[word] for word in all_words])
X = PCA(n_components=64).fit_transform(X)
```


## Cluster the words using HDBSCAN

To cluster the words, I used HDBSCAN, which is one of the most popular clustering algorithms. I want my clusters to be large enough to contain different concepts: for example, I want "car" and "bicycle" to be in the same cluster, despite one having a motor and the other not, they are both vehicles and I could identify a direction from motorized to non-motorized vehicles.

I also want to have a large number of clusters so I can find many different directions.

I tried different parameters, and I found that the following worked well.

```python
hac = hdbscan.HDBSCAN(min_cluster_size=5, metric='precomputed', max_cluster_size=50, min_samples=2, cluster_selection_method="leaf")
# Convert similarity to distance
similarity_matrix = cosine_similarity(X)
distance_matrix = similarity_matrix
distance_matrix = 1 - distance_matrix  # Convert similarity to distance
distance_matrix = np.clip(distance_matrix, 0, 1)  # Ensure values are in [0, 1]
distance_matrix = distance_matrix.astype(np.float64)  # Convert to float64 for HDBSCAN
# Fit the model
print("Fitting HDBSCAN model...")
labels = hac.fit_predict(distance_matrix)
# Print the number of clusters found
print(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")


# make the clusters
clusters = {}
for cluster_id in set(labels):
    if cluster_id == -1:
        continue  # Skip noise points
    cluster_words = [all_words[i] for i in range(len(all_words)) if labels[i] == cluster_id]
    clusters[cluster_id] = cluster_words

```

```python
# print the clusters
for cluster_id, data in enhanced_clusters.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Words: {', '.join(data)}")
```

    Cluster 0:
      Words: TV, broadcast, media, television, video
    Cluster 1:
      Words: butler, countess, king, princess, queen
    Cluster 2:
      Words: blizzard, cyclone, earthquake, hurricane, rain, rainstorm, rainy, sleet, snow, snowstorm, storm, thunderstorm, tsunami, typhoon, weather, winter
    Cluster 3:
      Words: diner, dining, restaurant, waiter, waitress
    Cluster 4:
      Words: demon, god, hellcat, temptress, thug, vixen
    Cluster 5:
      Words: blade, gun, hacksaw, handgun, handsaw, knife, mattock, pistol, pliers, rifle, scissors, screwdriver, weeder
    Cluster 6:
      Words: freight, rail, streetcar, tram, transit, transport, trolley
    Cluster 7:
      Words: SUV, bus, car, minibus, motorcycle, scooter, taxi, truck, van, vehicle
    Cluster 8:
      Words: availability, creation, development, expansion, growth, implementation, integration, standardization, transformation, utilization
    Cluster 9:
      Words: adjective, adverb, comma, hyphenation, neologism, noun, phrase, pronoun, punctuation, semicolon, synonym, verb
    Cluster 10:
      Words: colonialism, counterterrorism, expansionism, insurrection, military, terror, terrorism, war
    Cluster 11:
      Words: anesthesiologist, counselor, dentist, doctor, nurse, ophthalmologist, pharmacist, psychiatrist, psychologist, therapist
    Cluster 12:
      Words: aluminum, coal, copper, gas, hydrocarbon, methane, mining, ore, steel
    Cluster 13:
      Words: casement, coil, gripper, hexagon, parallelogram, plier, spool, valance
    Cluster 14:
      Words: aircraft, airplane, helicopter, jet, plane
    Cluster 15:
      Words: brake, footrest, gearshift, handlebar, headrest
    Cluster 16:
      Words: abdomen, ankle, bone, chest, chin, clavicle, elbow, eyebrow, finger, forearm, forehead, heel, hip, jaw, knee, leg, neck, nose, pinkie, rib, shin, shoulder, thigh, thumb, toe, toenail, torso, wrist
    Cluster 17:
      Words: bangle, bracelet, brooch, earring, earrings, figurine, necklace, ornament, pendant, porcelain, vase
    Cluster 18:
      Words: deodorant, lotion, mascara, shampoo, washcloth
    Cluster 19:
      Words: bladder, gland, intestine, liver, pancreas
    Cluster 20:
      Words: area, east, north, south, southeast, west, western
    Cluster 21:
      Words: armoire, banquette, bookcase, chaise, credenza, daybed, divan, dresser, footstool, futon, mattress, ottoman, recliner, sideboard, sofa
    Cluster 22:
      Words: carport, lanai, patio, porch, sunroom
    Cluster 23:
      Words: aunt, baby, boy, boyfriend, brother, cousin, dad, daddy, daughter, family, father, friend, girl, girlfriend, godmother, grandchild, granddaughter, grandfather, grandma, grandmom, grandmother, grandpa, grandson, husband, mama, mom, mother, nephew, niece, papa, roommate, sister, son, stepdaughter, stepmother, stepson, teenager, uncle, widow, wife, woman
    Cluster 24:
      Words: capitalism, democracy, ideology, socialism, socialist
    Cluster 25:
      Words: amazement, anger, anguish, anxiety, arrogance, bafflement, commitment, compassion, desire, disappointment, disgust, enthusiasm, exasperation, frustration, generosity, glee, gratitude, grief, hatred, heartache, heterosexual, homosexual, homosexuality, ignorance, joy, outrage, passion, prejudice, racism, regret, religion, reluctance, sadness, sorrow, sympathy, willingness
    Cluster 26:
      Words: custody, detention, jail, parole, prison, prosecution, sentence, sentencing
    Cluster 27:
      Words: boolean, browser, charset, desktop, initialize, integer, interface, login, modem, parser, plugin, postfix, router, server, software, subroutine, webmail
    Cluster 28:
      Words: average, decline, doubling, drop, half, increase, percent, percentage, rate, rise, total
    Cluster 29:
      Words: afternoon, day, end, fifth, first, last, leading, major, month, morning, night, round, second, set, today, week, weekend, will
    Cluster 30:
      Words: church, clergyman, nun, pastor, priest
    Cluster 31:
      Words: company, distributor, industry, market, subsidiary, supplier
    Cluster 32:
      Words: agreement, decision, legislation, partnership, policy, proposal
    Cluster 33:
      Words: apartment, basement, bathroom, bedroom, bungalow, chapel, chateau, downstairs, hallway, house, inn, mansion, monastery, palace, room, shrine, temple, townhouse, upstairs, villa
    Cluster 34:
      Words: debt, lender, lending, mortgage, pension
    Cluster 35:
      Words: assertion, implication, question, rationale, reason, reasoning, suggestion, theory
    Cluster 36:
      Words: athlete, athletics, baseball, basketball, boxer, champion, championship, coach, defeat, finisher, football, game, graduate, gymnast, gymnastics, hockey, hurdler, junior, league, runner, soccer, sociology, softball, sprinter, teammate, tennis, volleyball, win
    Cluster 37:
      Words: anybody, anyone, anything, everybody, going, maybe, nobody, nothing, somebody, thought, want
    Cluster 38:
      Words: ass, crap, damn, fuck, shit
    Cluster 39:
      Words: anthropology, biology, mathematics, neurobiologist, science
    Cluster 40:
      Words: dishwasher, dryer, heater, oven, refrigerator, stove, washer
    Cluster 41:
      Words: colt, filly, foal, gelding, heifer, mare, stallion
    Cluster 42:
      Words: expert, professor, researcher, scholar, scientist, technologist
    Cluster 43:
      Words: administrator, adviser, aide, assistant, attorney, chairman, chairperson, chief, columnist, correspondent, counsel, deputy, director, executive, journalist, lawyer, manager, officer, president, representative, secretary, spokesman, supervisor, vice, writer
    Cluster 44:
      Words: city, congressman, governor, lawmaker, legislator, mayor, minister, senator
    Cluster 45:
      Words: basin, canal, creek, estuary, lake, marsh, reservoir, river, shoreline, tributary, wetland
    Cluster 46:
      Words: barge, boat, canoe, catamaran, causeway, crewmen, dinghy, dory, dredger, ferryboat, freighter, frigate, houseboat, kayak, ketch, motorboat, pier, rowboat, sailboat, sampan, schooner, scow, ship, speedboat, tugboat, vessel, wharf, yacht, yawl
    Cluster 47:
      Words: blue, orange, pink, purple, red, tangerine, white, yellow
    Cluster 48:
      Words: ale, beer, brandy, drink, pinot, vodka, whiskey, wine
    Cluster 49:
      Words: anchovy, blackfish, carp, clam, cod, crab, crayfish, eel, fish, fishery, grouper, hake, halibut, herring, kingfish, lobster, mussel, oyster, salmon, scallops, seabass, seafood, shrimp, sprat, squid, sturgeon, swordfish, trout, tuna
    Cluster 50:
      Words: beetle, caterpillar, insect, moth, wasp, yellowjacket
    Cluster 51:
      Words: cat, dog, kitten, mutt, pug, pup, puppy, tabby
    Cluster 52:
      Words: alligator, anteater, antelope, bird, bobcat, cheetah, cougar, coyote, deer, dolphin, elephant, elk, frog, giraffe, gorilla, heron, lemur, leopard, lizard, lynx, mallard, manatee, moose, opossum, orangutan, osprey, otter, owl, panther, pelican, porpoise, raccoon, rhinoceros, sparrow, squirrel, tiger, toad, tortoise, turtle, whale, wolf
    Cluster 53:
      Words: begonia, crocus, cyclamen, dahlia, dogwood, flower, foxglove, geranium, lily, orchid, peony, tulip
    Cluster 54:
      Words: beech, birch, fir, larch, oak, sycamore
    Cluster 55:
      Words: corn, maize, rice, sorghum, wheat
    Cluster 56:
      Words: coleslaw, lasagna, meatloaf, pasta, polenta, quiche, ravioli, salad
    Cluster 57:
      Words: artichoke, asparagus, basil, broccoli, cabbage, cauliflower, celeriac, celery, chard, chive, chives, cilantro, cucumber, daikon, dill, eggplant, endive, fennel, garlic, jicama, kale, kohlrabi, melon, onion, oregano, parsley, radish, romaine, rosemary, scallion, shallot, tarragon, thyme, tomatillo, watercress, zucchini

## Cluster the directions within each cluster and only keep the relevant ones

This is the crux of the method. I want to find the clusters of directions within each cluster. So I calculated the pairwise difference between all words in the cluster, and then cluster the resulting vectors using HDBSCAN again.

The resulting clusters will have most of the clusters with the same words. For example:
- "north" > "city"
- "north" > "country"
- "north" > "state"

These elements have a similar direction but they are not relevant for our goal. They capture that "city", "country" and "state" are pretty close to each other in the embedding space, but they don't capture a meaningful direction.

To ensure diverse relationships within a direction cluster, I filtered it to only keep pairs where **all the words were unique**.

Then I only kept the clusters that have **at least 2 pairs** so we can do an average direction.

Finally, I define two metrics with their thresholds to filter the clusters:

1. **Intra-cluster coherence**: This measures how similar the directions in a cluster are to each other.
2. **Word-level coherence**: This measures whether the starting and ending words of the pairs form their own coherent groups.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

INTRA_CLUSTER_THRESHOLD = 0.3
WORD_LEVEL_THRESHOLD = 0.3

def find_relevant_clusters(cluster_id):
    all_words = enhanced_clusters[cluster_id]
    all_diff_vectors = []
    for w1 in all_words:
        for w2 in all_words:
            if w1 != w2:
                diff_vector = wv[w1] - wv[w2]
                all_diff_vectors.append(
                    {
                        "word1": w1,
                        "word2": w2,
                        "vector": diff_vector 
                    }
                )

    # do dbscan clustering with all_diff_vectors
    hac = hdbscan.HDBSCAN(min_cluster_size=10, metric='precomputed', max_cluster_size=50, min_samples=2, cluster_selection_method="leaf")
    similarity_matrix = cosine_similarity([vec["vector"] for vec in all_diff_vectors])
    distance_matrix = similarity_matrix
    distance_matrix = 1 - distance_matrix  # Convert similarity to distance
    distance_matrix = np.clip(distance_matrix, 0, 1)  # Ensure values are in [0, 1]
    distance_matrix = distance_matrix.astype(np.float64)  # Convert to float64 for HDBSCAN

    hac.fit(distance_matrix)
    labels = hac.labels_
    cur_clusters = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip noise points
        cur_clusters[cluster_id] = []
        for i, label in enumerate(labels):
            if label == cluster_id:
                cur_clusters[cluster_id].append((all_diff_vectors[i]["word1"], all_diff_vectors[i]["word2"]))


    # filter unique words
    for cluster_id, data in cur_clusters.items():
        new_data = []
        seen_words = set()
        for w1, w2 in data:
            if w1 not in seen_words and w2 not in seen_words:
                new_data.append((w1, w2))
                seen_words.add(w1)
                seen_words.add(w2)
        cur_clusters[cluster_id] = new_data

    # filter clusters with less than 2 pairs
    cur_clusters = {k: v for k, v in cur_clusters.items() if len(v) > 2}


    def calculate_intra_cluster_coherence(cluster_pairs, model):
        """
        Calculates how similar the relationship vectors in a cluster are to each other.
        A score close to 1.0 means a very consistent relationship.
        """
        diff_vectors = []
        for word1, word2 in cluster_pairs:
            if word1 in model and word2 in model:
                diff_vectors.append(model[word2] - model[word1])

        if len(diff_vectors) < 2:
            return 0.0 # Not enough vectors to compare

        # Calculate the centroid (average vector)
        centroid = np.mean(diff_vectors, axis=0).reshape(1, -1)

        # Calculate cosine similarity of each vector to the centroid
        similarities = cosine_similarity(diff_vectors, centroid)

        return np.mean(similarities)


    def calculate_word_level_coherence(cluster_pairs, model):
        """
        Calculates if the start words and end words form their own coherent groups.
        Returns two scores: (start_word_coherence, end_word_coherence).
        Scores close to 1.0 are best.
        """
        words_a = [pair[0] for pair in cluster_pairs]
        words_b = [pair[1] for pair in cluster_pairs]

        def get_internal_coherence(words, model):
            vectors = []
            for word in words:
                if word in model:
                    vectors.append(model[word])

            if len(vectors) < 2:
                return 0.0 # Not enough words to compare

            # Calculate average pairwise cosine similarity
            pair_similarities = []
            for v1, v2 in combinations(vectors, 2):
                pair_similarities.append(cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0])
            
            if not pair_similarities:
                return 0.0
                
            return np.mean(pair_similarities)

        coherence_a = get_internal_coherence(words_a, model)
        coherence_b = get_internal_coherence(words_b, model)

        return coherence_a, coherence_b



    scored_clusters = {}
    for cluster_id, pairs in cur_clusters.items():
        score1 = calculate_intra_cluster_coherence(pairs, wv)
        score2a, score2b = calculate_word_level_coherence(pairs, wv)
        scored_clusters[cluster_id] = {
            "pairs": pairs,
            "intra_cluster_coherence": score1,
            "word_level_coherence": (score2a, score2b)
        }

    relevant_clusters = {}
    for cluster_id, scores in scored_clusters.items():
        is_relevant = (
            scores["intra_cluster_coherence"] > INTRA_CLUSTER_THRESHOLD and
            scores["word_level_coherence"][0] > WORD_LEVEL_THRESHOLD and
            scores["word_level_coherence"][1] > WORD_LEVEL_THRESHOLD
        )

        if is_relevant:
            relevant_clusters[cluster_id] = scores

    return relevant_clusters
```

```python
import json
all_relevant_clusters = []
for i in tqdm(range(len(enhanced_clusters))):
    out = find_relevant_clusters(i)
    all_relevant_clusters.append(out)
```

    100%|██████████| 58/58 [00:01<00:00, 35.18it/s]


## Let's test it out!

Let's take the pairs from the cluster where it goes from one gender to another.

I compute the mean vector of the pairs, and then I add it to a word embedding to find similar words.
```python
# female to male relative
pairs =  [
            [
                "aunt",
                "brother"
            ],
            [
                "grandma",
                "daughter"
            ],
            [
                "grandmother",
                "nephew"
            ]
        ]
vecs = [wv[pair[1]] - wv[pair[0]] for pair in pairs]
mean_vec = np.mean(vecs, axis=0)
word = "daughter"
wv.similar_by_vector(wv[word] + mean_vec)
```




    [('son', 0.8330507278442383),
     ('daughter', 0.8292535543441772),
     ('nephew', 0.7205858826637268),
     ('brother', 0.7185276746749878),
     ('eldest_son', 0.6810173392295837),
     ('sons', 0.6789340972900391),
     ('father', 0.6786629557609558),
     ('eldest_daughter', 0.6753790974617004),
     ('younger_brother', 0.6528493762016296),
     ('daughters', 0.6517931222915649)]


We can see that the most similar embedding to "daughter" + vec_female_to_male is "son", and most of the other results are also related to male family members.

Let's compare with the similar words to "daughter" without the vector addition:

```python
wv.similar_by_word(word)
```




    [('mother', 0.8706234097480774),
     ('niece', 0.8637570738792419),
     ('granddaughter', 0.8516312837600708),
     ('son', 0.8468296527862549),
     ('daughters', 0.8136500716209412),
     ('eldest_daughter', 0.8052166700363159),
     ('sister', 0.7814769744873047),
     ('stepdaughter', 0.7707852721214294),
     ('wife', 0.7662219405174255),
     ('grandmother', 0.7483130097389221)]


The results are quite different! We correctly identified a direction vector in the embedding space that captures the female-to-male relationship!



## Attributing a meaning to the clusters

Finally, I used a LLM to interpret the clusters and find their meaning. I will use Google Gemini API to do that.

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from google import genai
import os
from pydantic import BaseModel, Field

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-2.5-pro"
```


```python
class Response(BaseModel):
    thinking: str = Field(..., description="Your reasoning process")
    final_answer: str = Field(..., description="Your final answer to the question")
    
prompt_template = """
I've calculated the difference between the word embedding of a large corpus and now I have these pairs that have a similar direction. Help me find some meaning to the directions of this cluster.
Your final answer should be a very short description of the cluster direction.
Their may be outliers in the pairs, so focus on the most common direction.
If you find that the pairs are not related, you can say "no relation" in your final answer.

Cluster directions:
{pairs}
""".strip()

def format_prompt(pairs):
    pairs_str = "\n".join([f"Pair {idx+1}: {pair[0]} -> {pair[1]}" for idx, pair in enumerate(pairs)])
    return prompt_template.format(pairs=pairs_str)

def get_cluster_meaning(pairs):
    prompt = format_prompt(pairs)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config = {
            "response_mime_type": "application/json",
            "response_schema" : Response,
            "system_instruction": "You are a helpful assistant expert in understanding word relationships and clusters. Your task is to analyze the provided pairs of words and provide a concise description of the cluster direction.",
        }
    )
    return response
```


```python
def get_cluster_meaning_wrapper(cluster):
    pairs = cluster["pairs"]
    response = get_cluster_meaning(pairs)
    cluster["meaning"] = response.parsed.final_answer
    cluster["thinking"] = response.parsed.thinking
    return cluster
```


```python
from multiprocessing import Pool
from tqdm import tqdm
import json


with Pool(20) as p:
    all_relevant_clusters_with_meaning = list(tqdm(p.imap(get_cluster_meaning_wrapper, all_relevant_clusters), total=len(all_relevant_clusters)))

with open("all_relevant_clusters_with_meaning.json", "w") as f:
    json.dump(all_relevant_clusters_with_meaning, f, indent=4)

```

    100%|██████████| 90/90 [01:26<00:00,  1.04it/s]



```python
for cluster in all_relevant_clusters_with_meaning:
    print(f"{cluster['meaning']}")
```
```text
    Weather phenomenon to natural disaster
    From a specific storm to a related weather phenomenon
    From a negative female archetype to a male archetype
    Hand tool to firearm
    Weapon to hand tool
    General transport concepts to specific rail elements
    From a specific mode of transport to its general purpose.
    From private to commercial/public transport
    From larger to smaller vehicles
    From specific processes to broader concepts of growth
    From process to expansion
    From concept to execution
    Shift between grammatical concepts
    From parts of speech to punctuation.
    Warfare and political violence
    From specific military actions to broader concepts of state expansion and war.
    From one medical profession to another
    From therapeutic or psychological professions to medical professions
    From metals and minerals to fossil fuels
    Energy resource to mineral resource
    From a geometric shape or component to a physical object or tool.
    Object to its geometry or associated tool
    From central or large body parts to peripheral parts, extremities, or joints.
    Lower to upper body parts
    Jewelry to Decorative Object
    General object to type of jewelry
    Geographical locations and directions
    From one geographic direction to another
    Storage furniture to seating furniture
    From seating or sleeping furniture to storage furniture
    From kinship term to demographic term
    Person descriptor to family role
    Female to male relative
    To an older, female relative
    Romantic partner to family member
    Core emotion to a strong reaction
    From intense emotions to sadness
    From emotion to sexual orientation
    From sexual orientation to strong emotion
    From motivational state to emotional state
    From emotion to volition
    High-level system/application to low-level implementation/programming component.
    Low-level programming concepts to high-level applications and hardware.
    From quantitative measures to descriptions of change
    From a type of change to its measurement
    Ordinal concept to time period
    From a unit of time to an ordinal position
    From a general business system to a specific functional part
    From a business entity to its larger operational context
    From religious building to secular room
    From secular space to sacred/religious space
    From a type of building to a part of that building
    From a part of a building to a whole building
    From a proposition to its logical basis or justification.
    Academic to athletic
    From sports to academia
    General sports terms to specific sports or athletes
    From verbs or states to indefinite pronouns
    General pronoun to a common associated action or state
    From kitchen appliances to other major home appliances
    Major household appliances
    no relation
    Female/neutral to male equine
    From a support/administrative role to a writing/media role
    From a writing/journalism profession to a staff/support role
    Legal professional to support staff
    From general support/administrative roles to legal professionals.
    From leader to subordinate
    From a subordinate or managerial role to a leadership or executive role
    From a large body of water to an adjacent or subsidiary feature.
    From a smaller water feature to a larger body of water or basin
    From a watercraft to a docking structure
    From a water-related structure to a specific watercraft
    Small watercraft to large watercraft
    Primary color to secondary color
    Color relationships
    From one type of alcoholic beverage to another
    From one type of alcoholic beverage to another
    From one type of seafood to another
    General insect to a specific pest or stinging insect
    Specific insect to general insect
    From wild land mammals to aquatic or exotic animals
    General animal to North American mammal
    From one type of flower to another
    Specific flower to another flower
    Broadleaf tree to conifer tree
    From American/European dishes to Italian dishes
    Italian food to non-Italian food
    Vegetable to herb
    herbs to vegetables
```
## Let's test a few more clusters

### From American/European dishes to Italian dishes

I test on the word "cheese" to see if it finds Italian cheeses.

Surprisingly, it finds at the top "pasta" meaning that the direction is more toward pasta dishes than italian dishes. But it still finds more Italian-related cheeses with the direction vector while the similar_by_word method mostly returns french cheeses.

```python
pairs =  [

            
            [
                "coleslaw",
                "lasagna"
            ],
            [
                "meatloaf",
                "pasta"
            ],
            [
                "quiche",
                "polenta"
            ],
            [
                "salad",
                "ravioli"
            ]
        ]
vecs = [wv[pair[1]] - wv[pair[0]] for pair in pairs]
mean_vec = np.mean(vecs, axis=0)
word = "cheese"
wv.similar_by_vector(wv[word] + mean_vec)
```




    [('cheese', 0.8760064840316772),
     ('cheeses', 0.6886467933654785),
     ('pasta', 0.6770588159561157),
     ('mozzarella', 0.6748825907707214),
     ('ricotta', 0.6693215370178223),
     ('cheddar', 0.6451700329780579),
     ('goat_cheese', 0.6403456330299377),
     ('mozzarella_cheese', 0.6397900581359863),
     ('Cheese', 0.6308974623680115),
     ('Mozzarella_cheese', 0.6121116280555725)]




```python
wv.similar_by_word(word)
```




    [('cheeses', 0.7788999676704407),
     ('cheddar', 0.7627597451210022),
     ('goat_cheese', 0.7297402024269104),
     ('Cheese', 0.7286962270736694),
     ('cheddar_cheese', 0.725513756275177),
     ('Cheddar_cheese', 0.6943708658218384),
     ('mozzarella', 0.6805710792541504),
     ('cheddar_cheeses', 0.6694672107696533),
     ('Camembert', 0.6623162031173706),
     ('gruyere', 0.6615148186683655)]


### Weather phenomenon to natural disaster

Here we test the word "wind" to see if it finds a natural disaster related to wind.

It correctly finds "cyclone", "hurricane", "typhoon" and "storm" as the most similar words, which are all related to wind phenomena. 

```python
pairs =  [
            [
                "blizzard",
                "cyclone"
            ],
            [
                "rain",
                "earthquake"
            ],
            [
                "rainstorm",
                "hurricane"
            ],
            [
                "rainy",
                "snowstorm"
            ],
            [
                "sleet",
                "storm"
            ],
            [
                "snow",
                "thunderstorm"
            ],
            [
                "weather",
                "tsunami"
            ],
            [
                "winter",
                "typhoon"
            ]
        ]

vecs = [wv[pair[1]] - wv[pair[0]] for pair in pairs]
mean_vec = np.mean(vecs, axis=0)
word = "wind"
wv.similar_by_vector(wv[word] + mean_vec)
```



    [('wind', 0.7509654760360718),
     ('cyclone', 0.6039589047431946),
     ('typhoon', 0.6032367944717407),
     ('winds', 0.5877481698989868),
     ('hurricane', 0.5860252380371094),
     ('storm', 0.5791043043136597),
     ('Wind', 0.5333985686302185),
     ('storms', 0.5293585062026978),
     ('tsunami', 0.529338538646698),
     ('Hurricane_Ike', 0.5260397791862488)]




```python
wv.similar_by_word(word)
```



    [('winds', 0.7204776406288147),
     ('Wind', 0.6752252578735352),
     ('paupers_graves_Xicotencatl', 0.6299487948417664),
     ('gusts', 0.5962637066841125),
     ('Winds', 0.594899594783783),
     ('breeze', 0.592047929763794),
     ('northwesterly_wind', 0.5902013182640076),
     ('wind_blows', 0.5894188284873962),
     ('breezes', 0.5883914232254028),
     ('southwesterly_wind', 0.5843492150306702)]



# Final thoughts

* In the tests, often the most similar word is the original word itself, I think this is because when I performed the clustering, I looked at the cosine distance. It tells the direction of the vector, but not its magnitude. Therefore, when adding the mean vector to the original word, I move in the direction of the cluster, but the magnitude of that move isn't controlled.

* Word2Vec is a relatively old model (2013), and it would be interesting to try this with a more recent model like GloVe. Maybe BERT or other transformer-based models would also work but the clusters would be more complex and harder to interpret.

* I could explore more memory-efficient methods for calculating and clustering the pairwise differences across the entire vocabulary.
