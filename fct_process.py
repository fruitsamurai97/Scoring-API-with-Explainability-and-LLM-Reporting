import re
import pandas as pd
import numpy as np


def extraction(exp_list):
    adjusted_features = []
    adjusted_thresholds = []
    values=[]
    for item in exp_list:
        condition, value = item
        split_condition = condition.split()
        
        try:
            # Attempt to determine if the initial segment is a numerical value
            float(split_condition[0])
            flag = True  # Flag is true if initial part is a number, indicating misplaced feature
        except ValueError:
            flag = False  # Flag remains false if initial part is not a number, indicating correct placement

        if flag:
            # Redefine extraction logic for cases where initial segment is numerical
            feature = split_condition[2]  # Feature is located after the relational symbol
            if '<=' in condition:
                threshold = '1'  # Set to '1' for upper bound in special range format
            elif '>' in condition:
                threshold = '0'  # Set to '0' for lower bound in reversed special format
        else:
            # Standard extraction logic for regular conditions
            feature = split_condition[0]
            if '<=' in condition or '<' in condition or '>' in condition:
                threshold = split_condition[2]  # Extract only the numeric part of the threshold

        adjusted_features.append(feature)
        adjusted_thresholds.append(threshold)
        values.append(value)
    return adjusted_features, adjusted_thresholds, values, exp_list

##############################

def extract_bounds(s):
    # Utiliser une expression régulière pour extraire les nombres flottants de la chaîne
    bounds = [float(num) for num in re.findall(r'-?[\d.]+', s)]
    # Vérifier si on a deux bornes numériques
    if len(bounds) == 2:
        # Identifier l'ordre des nombres en fonction des opérateurs utilisés
        if '<' in s or '<=' in s:
            return (bounds[0], bounds[1])
        else:
            return (bounds[1], bounds[0])
    else:
        # Retourner False si on n'a pas deux bornes
        return False