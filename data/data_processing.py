import pandas as pd
import re


def drop_na_fields(df):
    df.replace('–', pd.NA, inplace=True)
    df.replace('not supported', pd.NA, inplace=True)
    return df.dropna()


def onehot_encode_categorical_values(df, columns):
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df


def consolidate_color(color, primary_colors=None):
    color = color.lower()
    color = color.replace('metallic', '')
    color = color.strip()

    def replace_colors(original_color, replacement_color):
        for key, value in replacement_color.items():
            original_color = original_color.replace(key, value) if key in original_color else original_color
        return original_color

    conversion = {'mango': 'yellow', 'maroon': 'red', 'sea': 'blue', 'chalk': 'white', 'grey': 'gray', 'ash': 'gray',
                  'nero': 'black', 'graphite': 'gray', 'gun': 'gray'}
    color = replace_colors(color, conversion)
    if primary_colors is None:
        primary_colors = ['green', 'red', 'yellow', 'black', 'silver', 'white', 'gray', 'blue', 'brown']

    for primary in primary_colors:
        if primary in color:
            return primary

    return color


# Function to extract engine specs
def extract_engine_specs(engine_type):
    # Extract horsepower
    hp_match = re.search(r'(\d+\.\d+)HP', engine_type)
    hp = float(hp_match.group(1)) if hp_match else None

    # Extract displacement (L)
    displacement_match = re.search(r'(\d+\.\d+)L|(\d+\.\d+) Liter', engine_type)
    displacement = float(displacement_match.group(1)) if displacement_match and displacement_match.group(1) else (
        float(displacement_match.group(2)) if displacement_match and displacement_match.group(2) else None)

    # Extract number of cylinders and configuration
    cylinders_match = re.search(r'(\bV\d+|\bI\d+|\bStraight \d+|\b\d+ Cylinder)', engine_type)
    if cylinders_match:
        cylinders = re.search(r'\d+', cylinders_match.group(0))
        num_cylinders = int(cylinders.group(0)) if cylinders else None
        v_configuration = 'V' in cylinders_match.group(0)
    else:
        num_cylinders = None
        v_configuration = False

    # Check for electric motor
    if 'Electric Motor' in engine_type:
        hp = hp if hp else 0  # If hp is not available, default to 0
        displacement = displacement if displacement else 0  # Electric motors don't have displacement
        num_cylinders = num_cylinders if num_cylinders else 0  # Electric motors don't have cylinders
        v_configuration = False  # Electric motors don't have V configuration

    return hp, displacement, num_cylinders, v_configuration


def process_data(df):
    df = drop_na_fields(df)

    # Extract engine spec
    df[['horsepower', 'displacement', 'num_cylinders', 'v_configuration']] = df['engine'].apply(
        lambda x: pd.Series(extract_engine_specs(x)))
    df = df.drop(columns=['engine'])
    df = onehot_encode_categorical_values(df, ['brand', 'fuel_type', 'accident'])
    df = df.drop(columns=['ext_col', 'int_col', 'clean_title', 'transmission'])
    return df
