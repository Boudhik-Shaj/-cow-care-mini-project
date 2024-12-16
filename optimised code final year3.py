import numpy as np
import pandas as pd
from collections import Counter


df=pd.read_csv("training.csv")

#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis':{'mastitis':0,'blackleg':1,'bloat':2,'coccidiosis':3,'cryptosporidiosis':4,
'displaced_abomasum':5,'gut_worms':6,'listeriosis':7,'liver_fluke':8,'necrotic_enteritis':9,'peri_weaning_diarrhoea':10,
'rift_valley_fever':11,'rumen_acidosis':12,
'traumatic_reticulitis':13,'calf_diphtheria':14,'foot_rot':15,'foot_and_mouth':16,'ragwort_poisoning':17,'wooden_tongue':18,'infectious_bovine_rhinotracheitis':19,
'acetonaemia':20,'fatty_liver_syndrome':21,'calf_pneumonia':22,'schmallen_berg_virus':23,'trypanosomosis':24,'fog_fever':25}},inplace=True)

#List of the all symptoms is listed here in list sym.

sym=['anorexia','abdominal_pain','anaemia','abortions','acetone','aggression','arthrogyposis',
    'ankylosis','anxiety','bellowing','blood_loss','blood_poisoning','blisters','colic','Condemnation_of_livers',
    'coughing','depression','discomfort','dyspnea','dysentery','diarrhoea','dehydration','drooling',
    'dull','decreased_fertility','diffculty_breath','emaciation','encephalitis','fever','facial_paralysis','frothing_of_mouth',
    'frothing','gaseous_stomach','highly_diarrhoea','high_pulse_rate','high_temp','high_proportion','hyperaemia','hydrocephalus',
    'isolation_from_herd','infertility','intermittent_fever','jaundice','ketosis','loss_of_appetite','lameness',
    'lack_of-coordination','lethargy','lacrimation','milk_flakes','milk_watery','milk_clots',
    'mild_diarrhoea','moaning','mucosal_lesions','milk_fever','nausea','nasel_discharges','oedema',
    'pain','painful_tongue','pneumonia','photo_sensitization','quivering_lips','reduction_milk_vields','rapid_breathing',
    'rumenstasis','reduced_rumination','reduced_fertility','reduced_fat','reduces_feed_intake','raised_breathing','stomach_pain',
    'salivation','stillbirths','shallow_breathing','swollen_pharyngeal','swelling','saliva','swollen_tongue',
    'tachycardia','torticollis','udder_swelling','udder_heat','udder_hardeness','udder_redness','udder_pain','unwillingness_to_move',
    'ulcers','vomiting','weight_loss','weakness']

#List of Diseases (26 Cattle Diseases Mention) is listed in list disease.

disease=['mastitis','blackleg','bloat','coccidiosis','cryptosporidiosis',
        'displaced_abomasum','gut_worms','listeriosis','liver_fluke','necrotic_enteritis','peri_weaning_diarrhoea',
        ' rift_valley_fever','rumen_acidosis',
        'traumatic_reticulitis','calf_diphtheria','foot_rot','foot_and_mouth','ragwort_poisoning','wooden_tongue','infectious_bovine_rhinotracheitis',
'acetonaemia','fatty_liver_syndrome','calf_pneumonia','schmallen_berg_virus','trypanosomosis','fog_fever']


l2=[]
for i in range(0,len(sym)):
    l2.append(0)

#spliting into feature and target values
X = df[sym]
y = df["prognosis"]
np.ravel(y)

tr=pd.read_csv("testing.csv")
# replace in pandas for replacing the values
tr.replace({'prognosis':{'mastitis':0,'blackleg':1,'bloat':2,'coccidiosis':3,'cryptosporidiosis':4,
'displaced_abomasum':5,'gut_worms':6,'listeriosis':7,'liver_fluke':8,'necrotic_enteritis':9,'peri_weaning_diarrhoea':10,
'rift_valley_fever':11,'rumen_acidosis':12,
'traumatic_reticulitis':13,'calf_diphtheria':14,'foot_rot':15,'foot_and_mouth':16,'ragwort_poisoning':17,'wooden_tongue':18,'infectious_bovine_rhinotracheitis':19,
'acetonaemia':20,'fatty_liver_syndrome':21,'calf_pneumonia':22,'schmallen_berg_virus':23,'trypanosomosis':24,'fog_fever':25}},inplace=True)

# training Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X, y)

# training Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

# training Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, y)

# training KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn.fit(X, y)

# prediction
psymptoms = ['anorexia','dyspnea','jaundice','hyperaemia','ketosis']
for k in range(0,len(sym)):
    for z in psymptoms:
        if(z==sym[k]):
            l2[k]=1

inputtest = [l2]

dt_predict = dt.predict(inputtest)
dt_predicted=dt_predict[0]

rf_predict = rf.predict(inputtest)
rf_predicted=rf_predict[0]

inputtest = np.array(inputtest)

gnb_predict = gnb.predict(inputtest)
gnb_predicted=gnb_predict[0]

knn_predict = knn.predict(inputtest)
knn_predicted=knn_predict[0]

# comparing all the model and print common or else prediction of random forest


predictions = [dt_predicted, rf_predicted, gnb_predicted, knn_predicted]
print(predictions)

# Count the occurrences of each number in predictions
prediction_counts = Counter(predictions)

# Find the most prediction and its count
recurring_number, count = prediction_counts.most_common(1)[0]

# Print the most prediction
if count > 1:
    for a in range(0,len(disease)):
        if(recurring_number == a):
            rdisease=disease[a]
            print("Your cow is likely to have a ",disease[a])
else:
    if count == 1:
        Ranf = predictions[1]
        rdisease=disease[a]
        print("Your cow is likely to have a ",disease[Ranf]) # If there is no recurring number, print the second number in predictions


remedies = {
    'mastitis': 'Isolate the infected cow, use appropriate antibiotics, and maintain proper hygiene during milking.',
    'blackleg': 'Vaccinate cattle against blackleg, and provide prompt treatment with antibiotics if symptoms occur.',
    'bloat': 'Introduce anti-bloat medications, provide access to fresh water, and ensure proper grazing management.',
    'coccidiosis': 'Administer coccidiostats, maintain clean living conditions, and provide proper nutrition.',
    'cryptosporidiosis': 'Implement strict hygiene practices, provide supportive care, and administer appropriate medications.',
    'displaced_abomasum': 'Consult with a veterinarian for proper diagnosis and treatment, which may include surgery.',
    'gut_worms': 'Administer anthelmintic medications and practice good pasture management.',
    'listeriosis': 'Use antibiotics for treatment, improve hygiene, and provide proper nutrition.',
    'liver_fluke': 'Implement measures to control snail intermediate hosts and use anthelmintic medications.',
    'necrotic_enteritis': 'Administer antibiotics and improve overall flock management and hygiene.',
    'peri_weaning_diarrhoea': 'Implement proper nutrition, hygiene, and consider antibiotic treatment.',
    'rift_valley_fever': 'Vaccinate animals, practice vector control, and ensure proper hygiene.',
    'rumen_acidosis': 'Adjust the diet to prevent acidosis, provide access to clean water, and use buffers.',
    'traumatic_reticulitis': 'Consult with a veterinarian for proper diagnosis and treatment, which may include surgery.',
    'calf_diphtheria': 'Administer antibiotics, provide supportive care, and improve living conditions.',
    'foot_rot': 'Implement proper foot hygiene, provide hoof care, and use appropriate medications.',
    'foot_and_mouth': 'Quarantine affected animals, practice strict biosecurity, and consider vaccination.',
    'ragwort_poisoning': 'Remove ragwort plants from pastures, and provide supportive care to affected animals.',
    'wooden_tongue': 'Administer antibiotics, provide soft and easily chewable feed, and improve oral hygiene.',
    'infectious_bovine_rhinotracheitis': 'Vaccinate against IBR, isolate affected animals, and provide supportive care.',
    'acetonaemia': 'Adjust the diet, provide proper nutrition, and consult with a veterinarian for treatment.',
    'fatty_liver_syndrome': 'Improve diet and nutrition, provide proper care, and consult with a veterinarian for treatment.',
    'calf_pneumonia': 'Vaccinate against respiratory pathogens, provide proper ventilation, and administer antibiotics.',
    'schmallen_berg_virus': 'Vaccinate against SBV, practice vector control, and provide supportive care.',
    'trypanosomosis': 'Use trypanocidal medications, control vectors, and provide supportive care.',
    'fog_fever': 'Implement preventive measures, provide supportive care, and consult with a veterinarian for treatment.',
}

# disease_lower = disease.lower()
# if disease_lower in remedies:


print(f'As a basic remedy for {rdisease} please {remedies[rdisease]}')


