# -*- coding: utf-8 -*-
"""
Natural Language Generation
"""

import NLU


# Compose range queries
def range_query(relation, place, tmp_operator):
    sql = "query " + relation['name'] + " feed filter [.GeoData " + tmp_operator + " " + place + "] consume;"
    return sql


# Compose nearest neighbour queries
def nn_query(relation, place, num_neighbors):
    if(relation['GeoData'] == 'point'):
        sql = "query " + relation['name'] + " creatertree[GeoData] " + relation['name']\
            + " distancescan2 [" + place + ", " + num_neighbors + "] consume;"
    else:
        sql = "query " + relation['name'] + " creatertree[GeoData] " + relation['name']\
            + " distancescan3 [" + place + ", " + num_neighbors + "] consume;"
    return sql


# Compose spatial join queries
def spatial_join_query(spatial_relations, tmp_operator, index_point, index_region):
    sql = ''
    if tmp_operator == 'ininterior':
        sql = "query " + spatial_relations[index_point]['name'] + " feed {a} " + spatial_relations[index_region]['name']\
            + " feed {b} symmjoin [.GeoData_a ininterior ..GeoData_b] consume;"
    else:
        sql = "query " + spatial_relations[0]['name'] + " feed {a} " + spatial_relations[1]['name']\
            + " feed {b} symmjoin [.GeoData_a intersects1 ..GeoData_b] consume;"
    return sql


# Compose distance join queries
def distance_join_query(spatial_relations, max_distance):
    sql = "query " + spatial_relations[0]['name'] + " feed {a} " + spatial_relations[1]['name']\
        + " feed {b} symmjoin [distance(.GeoData_a, ..GeoData_b) <= " + max_distance + "] consume;"
    return sql


# Compose aggregation queries
def place_count_query(relation, place, tmp_operator):
    sql = "query " + relation['name'] + " feed filter [.GeoData " + tmp_operator + " " + place + "] count;"
    return sql


# Compose aggregation queries
def place_sum_query(relation, place):
    sql = "query " + relation['name'] + " feed extend [IntersectionArea: area(intersection1(.GeoData, "\
        + place + "))] sum[IntersectionArea];"
    return sql


# Compose aggregation queries
def place_max_query(relation, place):
    sql = "query " + relation['name'] + " feed extend [IntersectionArea: area(intersection1(.GeoData, "\
        + place + "))] sortby[IntersectionArea desc] head[1] consume;"
    return sql


# Compose aggregation queries
def aggregation_count_query(spatial_relations, tmp_operator, index_point, index_region):
    sql = ''
    if tmp_operator == 'ininterior':
        sql = "query " + spatial_relations[index_region]['name'] + " feed extend [Cnt: fun(t: TUPLE) "\
            + spatial_relations[index_point]['name'] + " feed filter [.GeoData ininterior attr(t, GeoData)] count] consume;"
    else:
        sql = "query " + spatial_relations[0]['name'] + " feed extend [Cnt: fun(t: TUPLE) "\
            + spatial_relations[1]['name'] + " feed filter [.GeoData intersects attr(t, GeoData)] count] consume;"
    return sql


# Compose aggregation queries
def aggregation_max_query(spatial_relations, tmp_operator, index_point, index_region):
    sql = ''
    if tmp_operator == 'ininterior':
        sql = "query " + spatial_relations[index_region]['name'] + " feed extend [Cnt: fun(t: TUPLE) "\
            + spatial_relations[index_point]['name'] + " feed filter [.GeoData ininterior attr(t, GeoData)] count] sortby[Cnt desc] head[1] consume;"
    else:
        sql = "query " + spatial_relations[0]['name'] + " feed extend [Cnt: fun(t: TUPLE) "\
            + spatial_relations[1]['name'] + " feed filter [.GeoData intersects attr(t, GeoData)] count] sortby[Cnt desc] head[1] consume;"
    return sql


# Determine how many relations in spatial_relations have non-null place attributes
def num_placeInRel(spatial_relations):
    if len(spatial_relations) == 0:
        num = 0
    elif len(spatial_relations) == 1:
        if spatial_relations[0]['place'] == '':
            num = 0
        else:
            num = 1
    elif len(spatial_relations) == 2:
        if spatial_relations[0]['place'] == '' and spatial_relations[1]['place'] == '':
            num = 0
        elif spatial_relations[0]['place'] != '' and spatial_relations[1]['place'] != '':
            num = 2
        else:
            num = 1
    else:
        num = 0  
    return num


# Returns the index of a relation in spatial_relations
def index_placeInRel(spatial_relations):
    for i in range(len(spatial_relations)):
        if spatial_relations[i]['place'] != '':
            index = i
            break
    return index


# Compose executable database queries
def secondo(sentence):
    # Obtain key semantic information
    query_type, spatial_relations, place, num_neighbors, max_distance = NLU.get_semantic_information(sentence)
    print(sentence)
    print("query_type: ", end='')
    print(query_type)
    print("spatial_relations: ", end='')
    print(spatial_relations)

    sql_secondo = ""

    # Process non-spatial queries
    if query_type == 'Non-spatial Query':
        spatial_relation = spatial_relations
        goal = place
        condition = num_neighbors

        print("goal: " + goal)
        print("condition: " + condition)
        print()

        if condition:
            sql_secondo = "SELECT " + goal + " FROM " + spatial_relation + " WHERE " + condition + ";"
        else:
            sql_secondo = "SELECT " + goal + " FROM " + spatial_relation + ";"
        return sql_secondo
    

    print("place: ", end='')
    print(place)
    print("neighbor_num: " + num_neighbors)
    print("max distance: " + str(max_distance))
    print()

    # Process basic queries
    if query_type in ['Basic-distance Query', 'Basic-direction Query', 'Basic-length Query', 'Basic-area Query']:
        if query_type in ['Basic-distance Query', 'Basic-direction Query']:
            # Determine the operator
            if query_type == 'Basic-distance Query':
                operator = 'distance'
            else:
                operator = 'direction'
            if len(place) == 2:
                if num_placeInRel(spatial_relations) == 0:
                    sql_secondo = "query " + operator + "(" + place[0] + ", " + place[1] + ");"
                elif num_placeInRel(spatial_relations) == 1:
                    if len(spatial_relations) == 1:
                        index = 0
                    else:
                        index = index_placeInRel(spatial_relations)
                    # Representation of the location
                    tmp_place = '(' + spatial_relations[index]['name'] + ' feed filter [.' + spatial_relations[index]['place_name_attr']\
                        + ' = "' + spatial_relations[index]['place'] + '"] extract[GeoData])'
                    if place[0] == spatial_relations[index]['place']:
                        sql_secondo = "query " + operator + "(" + place[1] + ", " + tmp_place + ");"
                    else:
                        sql_secondo = "query " + operator + "(" + place[0] + ", " + tmp_place + ");"
                elif num_placeInRel(spatial_relations) == 2:
                    tmp_place1 = '(' + spatial_relations[0]['name'] + ' feed filter [.' + spatial_relations[0]['place_name_attr']\
                        + ' = "' + spatial_relations[0]['place'] + '"] extract[GeoData])'
                    tmp_place2 = '(' + spatial_relations[1]['name'] + ' feed filter [.' + spatial_relations[1]['place_name_attr']\
                        + ' = "' + spatial_relations[1]['place'] + '"] extract[GeoData])'
                    sql_secondo = "query " + operator + "(" + tmp_place1 + ", " + tmp_place2 + ");"
                else:
                    print("[error]")
            else:
                print("[error]: The number of places should be 2.")
        else:
            # Determine the operator
            if query_type == 'Basic-length Query':
                operator = 'size'
            else:
                operator = 'area'
            if len(place) == 1:
                if num_placeInRel(spatial_relations) == 0:
                    sql_secondo = "query " + operator + "(" + place[0] + ");"
                elif num_placeInRel(spatial_relations) == 1:
                    index = index_placeInRel(spatial_relations)
                    # Representation of the location
                    tmp_place = '(' + spatial_relations[index]['name'] + ' feed filter [.' + spatial_relations[index]['place_name_attr']\
                        + ' = "' + spatial_relations[index]['place'] + '"] extract[GeoData])'
                    sql_secondo = "query " + operator + tmp_place + ";"
                else:
                    print("[error]")
            else:
                print("[error]: The number of places should be 1.")
        return sql_secondo

    relation_num = len(spatial_relations)
    if relation_num == 2:
        if spatial_relations[0]['place'] == '' and spatial_relations[1]['place'] == '':
            # Determine the operator
            tmp_operator = ''
            index_point = 0
            index_region = 0
            if spatial_relations[0]['GeoData'] == 'point':
                if spatial_relations[1]['GeoData'] == 'region':
                    tmp_operator = 'ininterior'
                    index_point = 0
                    index_region = 1
            elif spatial_relations[1]['GeoData'] == 'point':
                if spatial_relations[0]['GeoData'] == 'region':
                    tmp_operator = 'ininterior'
                    index_point = 1
                    index_region = 0
            else:
                tmp_operator = 'intersects'
            if query_type == "Spatial Join Query":
                if tmp_operator == '':
                    print("[error]: To judge the location relationship, spatial objects should be {line, region} x {line, region} || point x region.")
                    return sql_secondo
                sql_secondo = spatial_join_query(spatial_relations, tmp_operator, index_point, index_region)
            elif query_type == "Distance Join Query":
                sql_secondo = distance_join_query(spatial_relations, max_distance)
            elif query_type == "Aggregation-count Query":
                if tmp_operator == '':
                    print("[error]: To judge the location relationship, spatial objects should be {line, region} x {line, region} || point x region.")
                    return sql_secondo
                sql_secondo = aggregation_count_query(spatial_relations, tmp_operator, index_point, index_region)
            elif query_type == "Aggregation-max Query":
                if tmp_operator == '':
                    print("[error]: To judge the location relationship, spatial objects should be {line, region} x {line, region} || point x region.")
                    return sql_secondo
                sql_secondo = aggregation_max_query(spatial_relations, tmp_operator, index_point, index_region)
            else:
                print("[error]: The query type is incorrect.")
                return sql_secondo
        # If there are two locations, an error is reported.
        elif spatial_relations[0]['place'] != '' and spatial_relations[1]['place'] != '':
            print("[error]: The number of spatial relationships should be 1 or 2.")
            return sql_secondo
        else:
            # Determine the relation and location
            index_place = 0
            index_relation = 0
            if spatial_relations[0]['place'] != '':
                index_place = 0
                index_relation = 1
            else:
                index_place = 1
                index_relation = 0
            # Representation of the location
            tmp_place = '(' + spatial_relations[index_place]['name'] + ' feed filter [.' + spatial_relations[index_place]['place_name_attr']\
                + ' = "' + spatial_relations[index_place]['place'] + '"] extract[GeoData])'
            # Determine the operator
            tmp_operator = ''
            if spatial_relations[index_relation]['GeoData'] == 'point':
                if spatial_relations[index_place]['GeoData'] == 'region':
                    tmp_operator = 'ininterior'
            else:
                tmp_operator = 'intersects'
            if query_type == "Range Query":
                if tmp_operator == '':
                    print("[error]: To judge the location relationship, spatial objects should be {line, region} x {line, region} || point x region.")
                    return sql_secondo
                sql_secondo = range_query(spatial_relations[index_relation], tmp_place, tmp_operator)
            elif query_type == "Nearest Neighbor Query":
                sql_secondo = nn_query(spatial_relations[index_relation], tmp_place, num_neighbors)
            elif query_type == "Aggregation-count Query":
                if tmp_operator == '':
                    print("[error]: To judge the location relationship, spatial objects should be {line, region} x {line, region} || point x region.")
                    return sql_secondo
                sql_secondo = place_count_query(spatial_relations[index_relation], tmp_place, tmp_operator)
            elif query_type == "Aggregation-sum Query":
                if spatial_relations[index_relation]['GeoData'] == 'region' and spatial_relations[index_place]['GeoData'] == 'region':
                    sql_secondo = place_sum_query(spatial_relations[index_relation], tmp_place)
                else:
                    print("[error]: The Aggregation-sum query only considers the sum of areas where regions intersect.")
                    return sql_secondo
            elif query_type == "Aggregation-max Query":
                if spatial_relations[index_relation]['GeoData'] == 'region' and spatial_relations[index_place]['GeoData'] == 'region':
                    sql_secondo = place_max_query(spatial_relations[index_relation], tmp_place)
                else:
                    print("[error]: You can query the maximum area of intersecting regions.")
                    return sql_secondo
            else:
                print("[error]: The query type is incorrect.")
                return sql_secondo

    elif relation_num == 1:
        if(spatial_relations[0]['place'] == ''):
            if(len(place) > 0):
                # Determine the operator
                tmp_operator = ''
                if spatial_relations[0]['GeoData'] == 'point':
                    tmp_operator = 'ininterior'
                else:
                    tmp_operator = 'intersects'
                if query_type == "Range Query":
                    sql_secondo = range_query(spatial_relations[0], place[0], tmp_operator)
                elif query_type == "Nearest Neighbor Query":
                    sql_secondo = nn_query(spatial_relations[0], place[0], num_neighbors)
                elif query_type == "Aggregation-count Query":
                    sql_secondo = place_count_query(spatial_relations[0], place[0], tmp_operator)
                elif query_type == "Aggregation-sum Query":
                    sql_secondo = place_sum_query(spatial_relations[0], place[0])
                elif query_type == "Aggregation-max Query":
                    sql_secondo = place_max_query(spatial_relations[0], place[0])
                else:
                    print("[error]: The query type is incorrect.")
                    return sql_secondo
            else:
                print("[error]: If there is a spatial relationship, the number of places should also be 1.")
                return sql_secondo
        else:
            print("[error]: The number of spatial relationships should be 1 or 2.")
    else:
        print("[error]: The number of spatial relationships should be 1 or 2.")

    return sql_secondo
