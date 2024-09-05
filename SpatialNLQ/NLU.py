# -*- coding: utf-8 -*-
"""
Natural Language Understanding
"""

import sys
# Add the path of the module
sys.path.append('/home/lmy/anaconda3/lib/python3.8/site-packages')
import spacy
import re
import joblib
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import torch
import torch.nn as nn
import pickle
import numpy as np
# Show only warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


basic_path = '/home/lmy/secondo/Algebras/SpatialNLQ/'

labels = ['Range Query', 'Nearest Neighbor Query', 'Spatial Join Query', 'Distance Join Query', \
          'Aggregation-count Query', 'Aggregation-sum Query', 'Aggregation-max Query', \
            'Basic-distance Query', 'Basic-direction Query', 'Basic-length Query', 'Basic-area Query']

# Define LSTMCNN model
class LSTMCNN(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, num_classes, num_filters=100, kernel_sizes=[3,4,5]):
        super(LSTMCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.convolution_layers = nn.ModuleList([
            nn.Conv1d(in_channels=2*hidden_size, out_channels=num_filters, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_output, (h_n, c_n) = self.lstm(x)
        x = lstm_output.permute(0, 2, 1)
        convolution_outputs = []
        for convolution in self.convolution_layers:
            convolution_output = convolution(x)
            convolution_output = nn.functional.relu(convolution_output)
            max_pool_output = nn.functional.max_pool1d(convolution_output, kernel_size=convolution_output.size(2))
            convolution_outputs.append(max_pool_output)
        concatenated_tensor = torch.cat(convolution_outputs, dim=1)
        flatten_tensor = concatenated_tensor.view(concatenated_tensor.size(0), -1)
        dropout_output = self.dropout(flatten_tensor)
        logits = self.linear(dropout_output)
        return logits

# Load the model and related information
word_to_idx = torch.load(basic_path + 'save_models/word_to_idx.pth')
max_length = torch.load(basic_path + 'save_models/max_length.pth')
num_classes = len(labels)
model = LSTMCNN(len(word_to_idx), embedding_size=100, hidden_size=64, num_classes=num_classes, num_filters=100, kernel_sizes=[3,4,5])
model.load_state_dict(torch.load(basic_path + 'save_models/model.pth'))
model.eval()

# Predict the type of NLQ
def predict_type(text):
    vector = np.array([word_to_idx.get(word, 1) for word in text.split()] + [0]*(max_length-len(text.split())))
    vector_tensor = torch.LongTensor(vector).unsqueeze(0)
    with torch.no_grad():
        logits = model(vector_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
    return labels[predicted_class]


class ListNode:
    def __init__(self, key):
        self.key = key
        self.next = None

class HashTable:
    def __init__(self, initial_capacity=16, load_factor=0.75):
        self.capacity = initial_capacity
        self.load_factor = load_factor
        self.size = 0
        self.buckets = [None] * self.capacity

    # Compute the hash index for a given key
    def _hash(self, key):
        return hash(key) % self.capacity

    # Resize the hash table to a new capacity and rehash all existing elements
    def _resize(self, new_capacity):
        old_buckets = self.buckets
        self.buckets = [None] * new_capacity
        self.capacity = new_capacity
        self.size = 0

        for node in old_buckets:
            while node:
                self.insert(node.key)
                node = node.next

    # Insert a key into the hash table. Resize the table if the load factor is exceeded.
    def insert(self, key):
        if self.size / self.capacity >= self.load_factor:
            self._resize(self.capacity * 2)

        index = self._hash(key)
        node = self.buckets[index]

        if not node:
            self.buckets[index] = ListNode(key)
        else:
            while True:
                if node.key == key:
                    return  # Key already exists
                if not node.next:
                    break
                node = node.next
            node.next = ListNode(key)

        self.size += 1

    # Check if the hash table contains a key.
    def contains(self, key):
        index = self._hash(key)
        node = self.buckets[index]

        while node:
            if node.key == key:
                return True
            node = node.next

        return False

    def remove(self, key):
        index = self._hash(key)
        node = self.buckets[index]
        prev = None

        while node:
            if node.key == key:
                if prev:
                    prev.next = node.next
                else:
                    self.buckets[index] = node.next
                self.size -= 1
                if self.size / self.capacity < self.load_factor / 4 and self.capacity > 16:
                    self._resize(self.capacity // 2)
                return True
            prev = node
            node = node.next

        return False


_known = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90
}


# Convert English numbers to Arabic numbers
def spoken_word_to_number(n):
    n = n.lower().strip()
    if n in _known:
        return _known[n]
    else:
        inputWordArr = re.split('[ -]', n)
    # all single words are known
    assert len(inputWordArr) > 1
    # Check the pathological case where hundred is at the end or thousand is at end
    if inputWordArr[-1] == 'hundred':
        inputWordArr.append('zero')
        inputWordArr.append('zero')
    if inputWordArr[-1] == 'thousand':
        inputWordArr.append('zero')
        inputWordArr.append('zero')
        inputWordArr.append('zero')
    if inputWordArr[0] == 'hundred':
        inputWordArr.insert(0, 'one')
    if inputWordArr[0] == 'thousand':
        inputWordArr.insert(0, 'one')
    inputWordArr = [word for word in inputWordArr if word not in ['and', 'minus', 'negative']]
    currentPosition = 'unit'
    output = 0
    for word in reversed(inputWordArr):
        if currentPosition == 'unit':
            number = _known[word]
            output += number
            if number > 9:
                currentPosition = 'hundred'
            else:
                currentPosition = 'ten'
        elif currentPosition == 'ten':
            if word != 'hundred':
                number = _known[word]
                if number < 10:
                    output += number * 10
                else:
                    output += number
            # else: nothing special
            currentPosition = 'hundred'
        elif currentPosition == 'hundred':
            if word not in ['hundred', 'thousand']:
                number = _known[word]
                output += number * 100
                currentPosition = 'thousand'
            elif word == 'thousand':
                currentPosition = 'thousand'
            else:
                currentPosition = 'hundred'
        elif currentPosition == 'thousand':
            assert word != 'hundred'
            if word != 'thousand':
                number = _known[word]
                output += number * 1000
        else:
            assert "Can't be here" == None
    return (output)


# Replace punctuation with spaces
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile("[^0-9a-zA-Z\s-]")
    line = rule.sub(' ', line).strip()
    return line


# Conversion between singular and plural forms of English words
def get_addi_word(word):
    result = ''
    if word.endswith("ies"):
        result = word[0:-3] + 'y'
    elif word.endswith('ses'):
        result = word[0:-2]
    elif word.endswith('s'):
        result = word[0:-1]
    elif word.endswith('y'):
        result = word[0:-1] + 'ies'
    else:
        result = word + 's'
    return result


# Return the number of nearest neighbours
def get_neighbor_num(pos_neighbor, numbers, s):
    num_of_neighbor = 0
    if pos_neighbor != -1:
        num_of_neighbor = 1
        for i in numbers:
            pos = s.find(i)
            if pos < pos_neighbor:
                if i.isdigit():
                    num_of_neighbor = i
                else:
                    num_of_neighbor = spoken_word_to_number(i)
            elif pos == pos_neighbor + 8:
                if i.isdigit():
                    num_of_neighbor = i
                else:
                    num_of_neighbor = spoken_word_to_number(i)
    return num_of_neighbor


# Return the distance threshold
def get_max_distance(distance_number):
    max_distance = 0
    if len(distance_number) >= 1:
        tmpList = distance_number[0].split()
        # If it's a floating point number
        if tmpList[0].replace(".",'').isdigit():
            if "." in tmpList[0]:
                max_distance = float(tmpList[0])
            else:
                max_distance = int(tmpList[0])
        else:
            max_distance = spoken_word_to_number(tmpList[0])
        # Kilometer
        if "ilo" in tmpList[1]:
            max_distance = max_distance * 1000
    return max_distance


# Return the index of the tuple with the highest similarity score in noun_to_place
def get_max_score(noun_to_place):
    max = noun_to_place[0][1]
    max_id = 0
    for i in range(len(noun_to_place)):
        if noun_to_place[i][1] > max:
            max = noun_to_place[i][1]
            max_id = i
    return max_id


# Transform non-spatial natural language queries
# Return the target and condition of the query
def transform2SQL(words, attributes, attr_info):
    # Get comparative relationships
    df1 = pd.read_excel(basic_path + 'library/rel-comparison.xlsx')
    relCompare = []
    for index, row in df1.iterrows():
        entry = {
            'SQL': row['SQL'],
            'description': row.iloc[2:2 + row['description_num']].tolist()
        }
        relCompare.append(entry)
    # Get fixed relationships
    df2 = pd.read_excel(basic_path + 'library/rel-fixed.xlsx')
    relFixed = df2.to_dict(orient='records')
    # Get aggregation relationships
    df3 = pd.read_excel(basic_path + 'library/rel-aggregation.xlsx')
    relAggre = []
    for index, row in df3.iterrows():
        entry = {
            'SQL': row['SQL'],
            'description': row.iloc[2:2 + row['description_num']].tolist()
        }
        relAggre.append(entry)

    # Determine the number of attributes
    matching_attr_word = [word for word in words if word in attributes]
    # Replace the description with the name of the attribute
    matching_attr = []
    for attr in matching_attr_word:
        info = next((info for info in attr_info if info['name'] == attr or info['description'] == attr), None)
        matching_attr.append(info['name'])

    goal = ""
    condition = ""
    if len(matching_attr) == 1:
        attr = matching_attr[0]
        # Find relationships
        found_compare = False
        found_fixed = False
        
        for rel in relCompare:    
            # There is an attribute and comparison relationship
            if any(desc_word in words for desc_word in rel['description']):
                con1 = attr
                con2 = rel['SQL']
                index_desc = next((words.index(desc_word) for desc_word in rel['description'] if desc_word in words), None)
                if index_desc is not None and index_desc + 1 < len(words):
                    con3 = words[index_desc+1]
                    if not con3.isdigit():
                        con3 = f"'{con3}'"
                    condition = f"{con1} {con2} {con3}"
                    found_compare = True
                    break
                
        if not found_compare:
            for rel in relFixed:
                # There is an attribute and fixed relationship
                if rel['description1'] in words:
                    con1 = attr
                    con2 = rel['SQL1']
                    index_desc = words.index(rel['description1'])
                    if index_desc + 3 < len(words):
                        con3 = words[index_desc + 1]
                        con4 = rel['SQL2']
                        con5 = words[index_desc + 3]
                        condition = f"{con1} {con2} {con3} {con4} {con5}"
                        found_fixed = True
                        break

        if not any(desc_word in words for desc_word in [desc for rel in relAggre if rel['SQL'] == 'COUNT' for desc in rel['description']]):
            goal = "*"
        else:
            goal = "COUNT(*)"

        # There is an attribute and no comparison or fixed relationship
        if not found_compare and not found_fixed:
            goal = attr
            condition = ""

    # There are two attributes
    elif len(matching_attr) == 2:
        # Determine the target of the query
        goal = matching_attr[0]
        goal1 = ""
        for rel in relAggre:
            if any(desc_word in words for desc_word in rel['description']):
                goal1 = rel['SQL']
                break
        if goal1:
            goal = f"{goal1}({goal})"

        # Determine the conditions of the query
        attr = matching_attr[1]
        found_compare = False
        for rel in relCompare:    
            # There is a comparative relationship
            if any(desc_word in words for desc_word in rel['description']):
                con1 = attr
                con2 = rel['SQL']
                index_desc = next((words.index(desc_word) for desc_word in rel['description'] if desc_word in words), None)
                if index_desc is not None and index_desc + 1 < len(words):
                    con3 = words[index_desc+1]
                    if not con3.isdigit():
                        con3 = f"'{con3}'"
                    condition = f"{con1} {con2} {con3}"
                    found_compare = True
                    break
                
        if not found_compare:
            for rel in relFixed:
                # There is a fixed relationship
                if rel['description1'] in words:
                    con1 = attr
                    con2 = rel['SQL1']
                    index_desc = words.index(rel['description1'])
                    if index_desc + 3 < len(words):
                        con3 = words[index_desc + 1]
                        con4 = rel['SQL2']
                        con5 = words[index_desc + 3]
                        condition = f"{con1} {con2} {con3} {con4} {con5}"
                        break
    return goal, condition


# Extract key semantic information
def get_semantic_information(s): 
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(s)
    # words: all words
    words = []
    noun_low = []
    # noun_list: nouns
    noun_list = []
    for token in doc:
        # print(token.text, token.pos_)
        if not token.is_punct | token.is_space:
            words.append(token.orth_)
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                noun_list.append(token.text)
                noun_low.append(token.text.lower())
    
    # Extract spatial relations
    relations_file = pd.read_csv(basic_path + 'knowledge_base/spatial_relations.csv')
    relations_file['lower_name'] = relations_file['name'].apply(lambda x: x.lower().strip())
    tmp_noun = []
    for word in noun_low:
        tmp_noun.append(word)
        tmp_noun.append(get_addi_word(word))
    relations = relations_file[relations_file['lower_name'].isin(tmp_noun)]
    spatial_relations = relations.to_dict(orient='records')
    for relation in spatial_relations:
        relation['place'] = ''
        pos1 = tmp_noun.index(relation['lower_name'])
        pos = pos1 // 2
        del noun_list[pos]
        del tmp_noun[pos*2]
        del tmp_noun[pos*2]

    # Determine whether the query is spatial or non-spatial
    name_relations = [item['name'] for item in spatial_relations]
    attr_file = pd.read_excel(basic_path + 'library/attribute.xlsx')
    filtered_file = attr_file[attr_file['rName'].isin(name_relations)]
    attr_info = filtered_file.to_dict(orient='records')
    attributes = []
    attributes.extend(filtered_file['name'].unique())
    attributes.extend(filtered_file['description'].unique())
    # Non-spatial query
    if any(item in attributes for item in noun_list):
        cat_info = 'Non-spatial Query'
        spatial_relation = name_relations[0]
        goal, condition = transform2SQL(words, attributes, attr_info)
        # Consistent with what is returned by the spatial query
        tmp_result = ''
        return cat_info, spatial_relation, goal, condition, tmp_result

    # Predict the type of query
    cat_info = predict_type(s)

    # Extract the number of nearest neighbours and distance thresholds
    numbers = []
    distance_number = []
    for ii in doc.ents:
        if ii.label_ == "CARDINAL":
            numbers.append(str(ii))
        if ii.label_ == "QUANTITY":
            distance_number.append(str(ii))
    num_of_neighbor = 0
    pos_neighbor = s.find("nearest")
    if pos_neighbor != -1:
        num_of_neighbor = get_neighbor_num(pos_neighbor, numbers, s)
    else:
        pos_neighbor = s.find("closest")
        if pos_neighbor != -1:
            num_of_neighbor = get_neighbor_num(pos_neighbor, numbers, s)
        else:
            pos_neighbor = s.find("neighbor")
            if pos_neighbor != -1:
                num_of_neighbor = get_neighbor_num(pos_neighbor, numbers, s)
    max_distance = get_max_distance(distance_number)

    # Extract locations
    places_file = pd.read_csv(basic_path + 'knowledge_base/places.csv')
    place_list = places_file['name'].tolist()
    hash_table = HashTable()
    # Insert all places into the hash table
    for place in place_list:
        hash_table.insert(place)

    # noun_to_place1: exact matching locations, noun_to_place2: fuzzy matching locations
    noun_to_place1 = []
    noun_to_place2 = []
    
    # Exact match of locations
    for word in noun_list:
        if hash_table.contains(word):
            noun_to_place1.append(word)
    # Fuzzy match of locations
    if len(noun_to_place1) < 2:
        for word in noun_list:
            tmp = process.extractOne(word, place_list)
            if tmp[1] > 90:
                if len(noun_to_place1) == 1 and noun_to_place1[0] != tmp[0]:
                    noun_to_place2.append(tmp)
        ll = len(noun_to_place2)
        tmp_place2 = noun_to_place2
        if ll > 2:
            first_id = get_max_score(noun_to_place2)
            first_place = noun_to_place2[first_id][0]
            del noun_to_place2[first_id]
            second_id = get_max_score(noun_to_place2)
            second_place = noun_to_place2[second_id][0]
            noun_to_place2 = []
            noun_to_place2[0] = first_place
            noun_to_place2[1] = second_place
        elif ll == 2:
            if noun_to_place2[0][1] < noun_to_place2[1][1]:
                first_id = 1
                second_id = 0
            else:
                first_id = 0
                second_id = 1
            noun_to_place2 = []
            noun_to_place2.append(tmp_place2[first_id][0])
            noun_to_place2.append(tmp_place2[second_id][0])
        elif ll == 1:
            noun_to_place2 = []
            noun_to_place2.append(tmp_place2[0][0])

    place = []
    if len(noun_to_place1) == 0:
        place = noun_to_place2
    elif len(noun_to_place1) == 1:
        place = noun_to_place1
        if len(noun_to_place2) > 0:
            place.append(noun_to_place2[0])
    elif len(noun_to_place1) == 2:
        place = noun_to_place1
    else:
        place[0] = noun_to_place1[0]
        place[1] = noun_to_place1[1]
        
    if cat_info in ['Basic-distance Query', 'Basic-direction Query']:
        if len(place) == 2:
            t1 = places_file.loc[places_file['name'] == place[0]]['rel_id'].tolist()
            t2 = places_file.loc[places_file['name'] == place[1]]['rel_id'].tolist()
            if t1[0] > 0:
                flag = 0
                for relation in spatial_relations:
                    if t1[0] == relation['id']:
                        if len(spatial_relations) > 1:
                            relation['place'] = place[0]
                            flag = 1
                        break
                # Add a new relation to store the location
                if flag == 0:
                    add_rel = relations_file.loc[relations_file['id'] == t1[0]].to_dict(orient='records')
                    add_rel[0]['place'] = place[0]
                    spatial_relations.append(add_rel[0])
            if t2[0] > 0:
                flag = 0
                for relation in spatial_relations:
                    if t2[0] == relation['id']:
                        if len(spatial_relations) > 1:
                            relation['place'] = place[1]
                            flag = 1
                        break
                # Add a new relation to store the location
                if flag == 0:
                    add_rel = relations_file.loc[relations_file['id'] == t2[0]].to_dict(orient='records')
                    add_rel[0]['place'] = place[1]
                    spatial_relations.append(add_rel[0])
    
    else:
        if len(place) > 0:
            t = places_file.loc[places_file['name'] == place[0]]['rel_id'].tolist()
            tmp_place = place[0]
            if t[0] == 0:
                place = []
                place.append(tmp_place)
            else:
                flag = 0
                for relation in spatial_relations:
                    if t[0] == relation['id']:
                        if len(spatial_relations) > 1:
                            relation['place'] = place[0]
                            flag = 1
                        break
                # Add a new relation to store the location
                if flag == 0:
                    add_rel = relations_file.loc[relations_file['id'] == t[0]].to_dict(orient='records')
                    add_rel[0]['place'] = place[0]
                    spatial_relations.append(add_rel[0])

    return cat_info, spatial_relations, place, str(num_of_neighbor), str(max_distance)
