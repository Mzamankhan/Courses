'''
CIS 572 - Machine Learning - Winter 2017
Class Project - Visualizing data
Creating scatter plots: Features vs. SalePrice
Created on Feb 17, 2017

@author: Manujinda Wathugala
'''

import os

import matplotlib.pyplot as plt
import pandas as pd


# Dictionaries to convert nominal data to numeric values so that
# it is friendly for plotting
MSZoning = {'RL': 40, 'RM': 50, 'RH': 30, 'FV': 20, 'C (all)': 10}
# LotFrontage = {'130': 240, '137': 260, '134': 250, '138': 270, '24': 400, '21': 390, '95': 1060, '87': 980, '120': 180, '121': 190, '122': 200, '124': 210, '128': 220, '129': 230, '59': 700, '58': 690, '55': 660, '54': 650, '57': 680, '56': 670, '51': 620, '50': 610, '53': 640, '52': 630, '313': 420, '115': 150, '114': 140, '88': 990, '116': 160, '111': 120, '110': 110, '112': 130, '82': 930, '83': 940, '80': 910, '81': 920, '86': 970, '118': 170, '84': 950, '85': 960, '108': 90, '109': 100, '102': 30, '103': 40, '100': 10, '101': 20, '106': 70, '107': 80, '104': 50, '105': 60, '39': 500, '38': 490, '33': 440, '32': 430, '30': 410, '37': 480, '36': 470, '35': 460, '34': 450, '60': 710, '61': 720, '62': 730, '63': 740, '64': 750, '65': 760, '66': 770, '67': 780, '68': 790, '69': 800, '174': 370, 'NA': 0, '182': 380, '99': 1100, '98': 1090, '168': 360, '91': 1020, '90': 1010, '93': 1040, '92': 1030, '160': 350, '94': 1050, '97': 1080, '96': 1070, '89': 1000, '150': 320, '153': 340, '152': 330, '48': 590, '49': 600, '46': 570, '47': 580, '44': 550, '45': 560, '42': 530, '43': 540, '40': 510, '41': 520, '144': 300, '140': 280, '141': 290, '149': 310, '77': 880, '76': 870, '75': 860, '74': 850, '73': 840, '72': 830, '71': 820, '70': 810, '79': 900, '78': 890}
Street = {'Grvl': 10, 'Pave': 20}
Alley = {'Grvl': 10, 'Pave': 20, 'NA': 0}
LotShape = {'IR1': 10, 'IR2': 20, 'IR3': 30, 'Reg': 40}
LandContour = {'Bnk': 10, 'Lvl': 40, 'HLS': 20, 'Low': 30}
Utilities = {'AllPub': 10, 'NoSeWa': 20}
LotConfig = {'CulDSac': 20, 'Corner': 10, 'Inside': 50, 'FR3': 40, 'FR2': 30}
LandSlope = {'Sev': 30, 'Gtl': 10, 'Mod': 20}
Neighborhood = {'IDOTRR': 100, 'Edwards': 80, 'BrkSide': 40, 'OldTown': 180, 'NoRidge': 160, 'Veenker': 250, 'Gilbert': 90, 'SWISU': 190, 'Blmngtn': 10, 'NridgHt': 170, 'NWAmes': 150, 'StoneBr': 230, 'Somerst': 220, 'MeadowV': 110, 'ClearCr': 50, 'SawyerW': 210, 'Sawyer': 200, 'CollgCr': 60, 'Crawfor': 70, 'BrDale': 30, 'Mitchel': 120, 'NPkVill': 140, 'Blueste': 20, 'Timber': 240, 'NAmes': 130}
Condition1 = {'PosN': 50, 'RRAe': 60, 'RRNe': 80, 'RRNn': 90, 'RRAn': 70, 'PosA': 40, 'Artery': 10, 'Feedr': 20, 'Norm': 30}
Condition2 = {'PosN': 50, 'RRAe': 60, 'RRNn': 80, 'RRAn': 70, 'PosA': 40, 'Artery': 10, 'Feedr': 20, 'Norm': 30}
BldgType = {'Duplex': 30, '2fmCon': 20, '1Fam': 10, 'TwnhsE': 50, 'Twnhs': 40}
HouseStyle = {'1.5Unf': 20, '2.5Fin': 40, '2Story': 60, 'SFoyer': 70, '1.5Fin': 10, '2.5Unf': 50, 'SLvl': 80, '1Story': 30}
RoofStyle = {'Hip': 40, 'Flat': 10, 'Gable': 20, 'Mansard': 50, 'Shed': 60, 'Gambrel': 30}
RoofMatl = {'Tar&Grv': 60, 'ClyTile': 10, 'Metal': 40, 'WdShake': 70, 'WdShngl': 80, 'Membran': 30, 'CompShg': 20, 'Roll': 50}
Exterior1st = {'Stone': 110, 'HdBoard': 70, 'BrkComm': 30, 'WdShing': 150, 'VinylSd': 130, 'CemntBd': 60, 'ImStucc': 80, 'Wd Sdng': 140, 'AsbShng': 10, 'AsphShn': 20, 'BrkFace': 40, 'Plywood': 100, 'MetalSd': 90, 'CBlock': 50, 'Stucco': 120}
Exterior2nd = {'Stone': 120, 'HdBoard': 70, 'VinylSd': 140, 'CmentBd': 60, 'Other': 100, 'Brk Cmn': 30, 'ImStucc': 80, 'Wd Sdng': 150, 'AsbShng': 10, 'AsphShn': 20, 'BrkFace': 40, 'Wd Shng': 160, 'Plywood': 110, 'MetalSd': 90, 'CBlock': 50, 'Stucco': 130}
MasVnrType = {'Stone': 40, 'NA': 0, 'None': 30, 'BrkFace': 20, 'BrkCmn': 10}
# MasVnrArea = {'216': 980, '768': 3010, '215': 970, '212': 960, '210': 950, '762': 2990, '452': 2320, '664': 2860, '766': 3000, '621': 2740, '218': 990, '219': 1000, '132': 330, '130': 320, '137': 360, '136': 350, '135': 340, '226': 1060, '95': 3220, '571': 2630, '138': 380, '225': 1050, '650': 2810, '24': 1150, '224': 1040, '22': 1010, '428': 2200, '223': 1030, '28': 1420, '450': 2300, '94': 3210, '289': 1500, '0': 10, '541': 2560, '340': 1820, '97': 3240, '342': 1830, '281': 1440, '1378': 370, '220': 1020, '285': 1460, '284': 1450, '287': 1480, '286': 1470, '673': 2880, '120': 250, '262': 1320, '122': 260, '123': 270, '125': 280, '126': 290, '127': 300, '128': 310, '268': 1350, '365': 1930, '348': 1850, '425': 2180, '57': 2620, '56': 2580, '51': 2490, '50': 2460, '53': 2530, '378': 1990, '63': 2750, '410': 2110, '921': 3190, '298': 1580, '299': 1590, '296': 1560, '297': 1570, '294': 1540, '295': 1550, '292': 1520, '293': 1530, '290': 1510, '376': 1980, '1129': 170, '456': 2330, '318': 1710, '594': 2680, '200': 860, '194': 840, '196': 850, '203': 880, '315': 1700, '192': 830, '115': 200, '114': 190, '117': 220, '116': 210, '274': 1390, '110': 140, '113': 180, '112': 160, '278': 1410, '205': 900, '399': 2070, '81': 3060, '119': 240, '250': 1230, '84': 3090, '204': 890, '796': 3040, '443': 2270, '207': 920, '415': 2130, '206': 910, '894': 3160, '491': 2450, '27': 1360, '651': 2820, '254': 1250, '368': 1950, '366': 1940, '420': 2150, '423': 2160, '255': 1260, '362': 1920, '424': 2170, '360': 1900, '361': 1910, '70': 2900, '309': 1660, '584': 2670, '448': 2280, '860': 3120, '412': 2120, '300': 1610, '442': 2260, '302': 1620, '530': 2540, '304': 1630, '305': 1640, '306': 1650, '245': 1190, '244': 1180, '108': 110, '109': 120, '240': 1160, '243': 1170, '370': 1960, '102': 50, '100': 30, '101': 40, '106': 100, '248': 1220, '104': 70, '105': 90, '380': 2010, '760': 2980, '38': 2000, '381': 2020, '32': 1720, '31': 1670, '30': 1600, '788': 3030, '247': 1210, '34': 1810, '391': 2050, '640': 2790, '246': 1200, '438': 2240, '436': 2230, '510': 2500, '513': 2510, '435': 2220, '432': 2210, '573': 2640, '459': 2340, '579': 2660, '338': 1800, '604': 2720, '335': 1770, '451': 2310, '337': 1790, '336': 1780, '576': 2650, '333': 1760, '60': 2690, '388': 2040, '258': 1280, '259': 1290, '64': 2780, '65': 2800, '66': 2840, '178': 740, '252': 1240, '176': 730, '175': 720, '174': 710, '256': 1270, '172': 700, '171': 690, '170': 680, '554': 2570, '344': 1840, '975': 3250, '603': 2710, '288': 1490, '387': 2030, 'NA': 0, '772': 3020, '88': 3140, '182': 770, '183': 780, '180': 760, '186': 800, '653': 2830, '184': 790, '312': 1690, '506': 2480, '188': 810, '1031': 60, '922': 3200, '202': 870, '630': 2760, '632': 2770, '310': 1680, '468': 2380, '464': 2360, '562': 2590, '466': 2370, '564': 2600, '567': 2610, '99': 3270, '98': 3260, '528': 2520, '168': 660, '169': 670, '228': 1070, '164': 620, '165': 630, '166': 640, '167': 650, '160': 570, '161': 590, '162': 600, '163': 610, '11': 130, '270': 1370, '189': 820, '14': 390, '16': 560, '660': 2850, '18': 750, '731': 2930, '272': 1380, '89': 3150, '500': 2470, '408': 2090, '54': 2550, '275': 1400, '151': 500, '150': 490, '153': 510, '92': 3180, '600': 2700, '154': 520, '157': 540, '156': 530, '748': 2950, '158': 550, '816': 3070, '36': 1890, '396': 2060, '82': 3080, '90': 3170, '238': 1140, '234': 1110, '236': 1120, '237': 1130, '230': 1080, '426': 2190, '232': 1090, '233': 1100, '375': 1970, '280': 1430, '48': 2420, '46': 2350, '86': 3110, '44': 2250, '45': 2290, '42': 2140, '40': 2080, '41': 2100, '1': 20, '320': 1730, '1115': 150, '324': 1740, '870': 3130, '328': 1750, '85': 3100, '146': 450, '147': 460, '144': 430, '145': 440, '142': 410, '143': 420, '140': 400, '1170': 230, '209': 940, '208': 930, '616': 2730, '148': 470, '149': 480, '76': 2970, '75': 2960, '74': 2940, '72': 2920, '68': 2890, '96': 3230, '481': 2440, '480': 2430, '263': 1330, '1047': 80, '80': 3050, '705': 2910, '1600': 580, '261': 1310, '472': 2390, '473': 2400, '260': 1300, '351': 1870, '350': 1860, '479': 2410, '67': 2870, '359': 1880, '266': 1340}
ExterQual = {'Fa': 20, 'Gd': 30, 'Ex': 10, 'TA': 40}
ExterCond = {'TA': 50, 'Fa': 20, 'Gd': 30, 'Ex': 10, 'Po': 40}
Foundation = {'Stone': 50, 'BrkTil': 10, 'Slab': 40, 'PConc': 30, 'Wood': 60, 'CBlock': 20}
BsmtQual = {'TA': 40, 'Fa': 20, 'Gd': 30, 'Ex': 10, 'NA': 0}
BsmtCond = {'TA': 40, 'Fa': 10, 'Gd': 20, 'Po': 30, 'NA': 0}
BsmtExposure = {'No': 40, 'NA': 0, 'Gd': 20, 'Mn': 30, 'Av': 10}
BsmtFinType1 = {'NA': 0, 'LwQ': 40, 'BLQ': 20, 'Unf': 60, 'GLQ': 30, 'Rec': 50, 'ALQ': 10}
BsmtFinType2 = {'NA': 0, 'LwQ': 40, 'BLQ': 20, 'Unf': 60, 'GLQ': 30, 'Rec': 50, 'ALQ': 10}
Heating = {'Floor': 10, 'GasA': 20, 'Grav': 40, 'Wall': 60, 'OthW': 50, 'GasW': 30}
HeatingQC = {'TA': 50, 'Fa': 20, 'Gd': 30, 'Ex': 10, 'Po': 40}
CentralAir = {'Y': 20, 'N': 10}
Electrical = {'FuseA': 10, 'FuseF': 20, 'Mix': 40, 'NA': 0, 'FuseP': 30, 'SBrkr': 50}
KitchenQual = {'Fa': 20, 'Gd': 30, 'Ex': 10, 'TA': 40}
Functional = {'Sev': 60, 'Min1': 30, 'Min2': 40, 'Typ': 70, 'Maj1': 10, 'Maj2': 20, 'Mod': 50}
FireplaceQu = {'NA': 0, 'Fa': 20, 'Gd': 30, 'Ex': 10, 'Po': 40, 'TA': 50}
GarageType = {'Basment': 30, 'CarPort': 50, 'NA': 0, 'Attchd': 20, 'BuiltIn': 40, '2Types': 10, 'Detchd': 60}
# GarageYrBlt = {'1948': 350, '1949': 360, '1942': 310, '1940': 290, '1941': 300, '1946': 330, '1947': 340, '1945': 320, '2010': 970, '1955': 420, '1954': 410, '1957': 440, '1956': 430, '1951': 380, '1950': 370, '1953': 400, '1952': 390, '1959': 460, '1958': 450, '1920': 90, '1921': 100, '1922': 110, '1923': 120, '1924': 130, '1925': 140, '1926': 150, '1927': 160, '1928': 170, '1929': 180, '1933': 220, '1932': 210, '1931': 200, '1930': 190, '1937': 260, '1936': 250, '1935': 240, '1934': 230, '1939': 280, '1938': 270, '1908': 30, '1906': 20, '1900': 10, '1986': 730, '1987': 740, '1984': 710, '1985': 720, '1982': 690, '1983': 700, '1980': 670, '1981': 680, 'NA': 0, '1988': 750, '1989': 760, '1918': 80, '1910': 40, '1915': 60, '1914': 50, '1916': 70, '1991': 780, '1990': 770, '1993': 800, '1992': 790, '1995': 820, '1994': 810, '1997': 840, '1996': 830, '1999': 860, '1998': 850, '1968': 550, '1969': 560, '1964': 510, '1965': 520, '1966': 530, '1967': 540, '1960': 470, '1961': 480, '1962': 490, '1963': 500, '1979': 660, '1978': 650, '1977': 640, '1976': 630, '1975': 620, '1974': 610, '1973': 600, '1972': 590, '1971': 580, '1970': 570, '2002': 890, '2003': 900, '2000': 870, '2001': 880, '2006': 930, '2007': 940, '2004': 910, '2005': 920, '2008': 950, '2009': 960}
GarageFinish = {'NA': 0, 'Fin': 10, 'RFn': 20, 'Unf': 30}
GarageQual = {'NA': 0, 'Fa': 20, 'Gd': 30, 'Ex': 10, 'Po': 40, 'TA': 50}
GarageCond = {'NA': 0, 'Fa': 20, 'Gd': 30, 'Ex': 10, 'Po': 40, 'TA': 50}
PavedDrive = {'Y': 30, 'P': 20, 'N': 10}
PoolQC = {'Fa': 20, 'Gd': 30, 'Ex': 10, 'NA': 0}
Fence = {'NA': 0, 'GdPrv': 10, 'MnWw': 40, 'MnPrv': 30, 'GdWo': 20}
MiscFeature = {'TenC': 40, 'NA': 0, 'Shed': 30, 'Gar2': 10, 'Othr': 20}
SaleType = {'Oth': 80, 'WD': 90, 'ConLI': 50, 'ConLD': 40, 'COD': 10, 'New': 70, 'ConLw': 60, 'CWD': 20, 'Con': 30}
SaleCondition = {'Partial': 60, 'Family': 40, 'Normal': 50, 'AdjLand': 20, 'Alloca': 30, 'Abnorml': 10}

MoSold = {'1': 10, '2': 20, '3': 30, '4': 40, '5': 50, '6': 60, '7': 70, '8': 80, '9': 90, '10': 100, '11': 110, '12': 120}
OverallQual = {'1': 10, '2': 20, '3': 30, '4': 40, '5': 50, '6': 60, '7': 70, '8': 80, '9': 90, '10': 100}
OverallCond = {'1': 10, '2': 20, '3': 30, '4': 40, '5': 50, '6': 60, '7': 70, '8': 80, '9': 90}
MSSubClass = {'160': 130, '70': 70, '40': 30, '75': 80, '45': 40, '190': 150, '80': 90, '50': 50, '20': 10, '85': 100, '120': 120, '180': 140, '90': 110, '60': 60, '30': 20}


converters = {'MSSubClass': lambda x: MSSubClass[x], 'OverallCond': lambda x: OverallCond[x], 'OverallQual': lambda x: OverallQual[x], 'MoSold': lambda x: MoSold[x], 'MSZoning': lambda x: MSZoning[x], 'LotFrontage': lambda x: 0 if x == 'NA' else int( x ), 'Street': lambda x: Street[x], 'Alley': lambda x: Alley[x], 'LotShape': lambda x: LotShape[x], 'LandContour': lambda x: LandContour[x], 'Utilities': lambda x: Utilities[x], 'LotConfig': lambda x: LotConfig[x], 'LandSlope': lambda x: LandSlope[x], 'Neighborhood': lambda x: Neighborhood[x], 'Condition1': lambda x: Condition1[x], 'Condition2': lambda x: Condition2[x], 'BldgType': lambda x: BldgType[x], 'HouseStyle': lambda x: HouseStyle[x], 'RoofStyle': lambda x: RoofStyle[x], 'RoofMatl': lambda x: RoofMatl[x], 'Exterior1st': lambda x: Exterior1st[x], 'Exterior2nd': lambda x: Exterior2nd[x], 'MasVnrType': lambda x: MasVnrType[x], 'MasVnrArea': lambda x:-100 if x == 'NA' else int( x ), 'ExterQual': lambda x: ExterQual[x], 'ExterCond': lambda x: ExterCond[x], 'Foundation': lambda x: Foundation[x], 'BsmtQual': lambda x: BsmtQual[x], 'BsmtCond': lambda x: BsmtCond[x], 'BsmtExposure': lambda x: BsmtExposure[x], 'BsmtFinType1': lambda x: BsmtFinType1[x], 'BsmtFinType2': lambda x: BsmtFinType2[x], 'Heating': lambda x: Heating[x], 'HeatingQC': lambda x: HeatingQC[x], 'CentralAir': lambda x: CentralAir[x], 'Electrical': lambda x: Electrical[x], 'KitchenQual': lambda x: KitchenQual[x], 'Functional': lambda x: Functional[x], 'FireplaceQu': lambda x: FireplaceQu[x], 'GarageType': lambda x: GarageType[x], 'GarageYrBlt': lambda x: 1890 if x == 'NA' else int( x ), 'GarageFinish': lambda x: GarageFinish[x], 'GarageQual': lambda x: GarageQual[x], 'GarageCond': lambda x: GarageCond[x], 'PavedDrive': lambda x: PavedDrive[x], 'PoolQC': lambda x: PoolQC[x], 'Fence': lambda x: Fence[x], 'MiscFeature': lambda x: MiscFeature[x], 'SaleType': lambda x: SaleType[x], 'SaleCondition': lambda x: SaleCondition[x]}


# All the nomilal attributes
nominal = ['YrSold', 'MoSold', 'OverallQual', 'OverallCond', 'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

house_data = pd.read_csv( 'train.csv', index_col = 0, na_filter = False, converters = converters )

# Create a folder to store the plots
plot_dir = os.path.join( os.getcwd(), 'scatter_plots' )
if not os.path.exists( plot_dir ):
    os.mkdir( plot_dir )

# For all the Features h
for h in list( house_data ):
# for h in ['YrSold']:
    title = 'Distribution of {} - {}'.format( h, '{}' )
    if h in nominal:
        title = title.format( 'Nominal' )

        # Group examples based on the values of the
        # feature and the SalePrice.
        grp = house_data.groupby( [h, 'SalePrice'] )

        # Maximum number of examples in any of the
        # groups
        m = ( max( grp.size() ) )

        # Jitter amout to add to the points so that
        # we can see a dot plot for each value of a
        # nominal feature.
        jit = 9

        # To accumulate the newly created set of points
        # with the jitter added
        x = []
        y = []

        fig, ax = plt.subplots()

        # For each group ( feature value, SalePrice)
        for g in grp.groups:

            # Number of examples in this group
            g_size = len( grp.get_group( g ) )

            # For each example in this group
            # Add some jitter to the x value
            for p in range( g_size ):
                y.append( g[1] )
                x.append( g[0] - ( p * jit ) / float( m ) )

        plt.scatter( x, y, s = 80, alpha = 0.4 )
    else:
        title = title.format( 'Numeric' )
        plt.scatter( house_data[h], house_data['SalePrice'], s = 80, alpha = 0.4 )

    plt.title( title )
    plt.ylabel( 'Sale Price' )
    plt.xlabel( '{}'.format( h ) );

    plt.savefig( os.path.join( plot_dir, '{}.png'.format( h ) ) )
    # plt.show()
    plt.close()
    plt.cla()
    plt.clf()

print ( 'Done' )
