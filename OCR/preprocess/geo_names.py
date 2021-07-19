import re

import utils

US_STATES = {
    'Alabama': {'AL','Ala'},
    'Alaska': {'AK'},
    'Arizona': {'AZ','Ariz'},
    'Arkansas': {'AR','Ark'},
    'California': {'CA','Calif'},
    'Colorado': {'CO','Colo'},
    'Connecticut': {'CT','Conn'},
    'Delaware': {'DE','Del'},
    'District-Of-Columbia':	{'DC'},
    'Florida': {'FL','Fla'},
    'Georgia': {'GA','Ga'},
    'Hawaii': {'HI','Hawaii'},
    'Idaho': {'ID','Idaho'},
    'Illinois': {'IL','Ill'},
    'Indiana': {'IN','Ind'},
    'Iowa': {'IA','Iowa'},
    'Kansas': {'KS','Kans'},
    'Kentucky': {'KY','Ky'},
    'Louisiana': {'LA','La'},
    'Maine': {'ME','Maine'},
    'Maryland': {'MD','Md'},
    'Massachusetts': {'MA','Mass'},
    'Michigan': {'MI','Mich'},
    'Minnesota': {'MN','Minn'},
    'Mississippi': {'MS','Miss'},
    'Missouri': {'MO','Mo'},
    'Montana': {'MT','Mont'},
    'Nebraska': {'NE','Nebr'},
    'Nevada': {'NV','Nev'},
    'New-Hampshire': {'NH'},
    'New-Jersey': {'NJ'},
    'New-Mexico': {'NM'},
    'New-York': {'NY'},
    'North-Carolina': {'NC'},
    'North-Dakota': {'ND'},
    'Ohio': {'OH','Ohio'},
    'Oklahoma': {'OK','Okla'},
    'Oregon': {'OR','Ore'},
    'Pennsylvania': {'PA','Pa'},
    'Rhode-Island': {'RI'},
    'South-Carolina': {'SC'},
    'South-Dakota': {'SD'},
    'Tennessee': {'TN','Tenn'},
    'Texas': {'TX','Tex'},
    'Utah': {'UT','Utah'},
    'Vermont': {'VT','Vt'},
    'Virginia': {'VA','Va'},
    'Washington': {'WA','Wash'},
    'West-Virginia': { 'WV','WVa'},
    'Wisconsin': {'WI','Wis'},
    'Wyoming': {'WY','Wyo'}
}

STATES_ALIASES = {}
for name, aa in US_STATES.items():
    for a in aa:
        for a_ in set([a, a.title(), a.upper(), a.lower(), name]):
            if a_ not in STATES_ALIASES:
                STATES_ALIASES[a_] = name

def normalize(name):
    """ Name normalization: clean and convert to canonical name. 
        Example: '@woRD1!+= <w>%o&r*(d2)?^.---[w~O]{$#rd3}' -> 'Word1-Word2-Word3'
    """
    name = re.sub(r"[#\.,;:!@\$%&\*\(\)\[\]\{\}\+=\?<>\\\/\|\^~]", '', name)
    name = '-'.join([w.title() for w in re.split(r'\W+', name)])
    return name

def find_closest_name(name, names):
    max_ratio, closest_name = 0, None
    for name_ in names:
        ratio = utils.levenshtein_ratio_and_distance(name, name_, ratio_calc=True)
        if ratio > max_ratio:
            max_ratio = ratio
            closest_name = name_
    return (max_ratio, closest_name)

def canonic_state_name(name, levensht_thresh=0.7):
    """ Get canonical state name.
        If it is not valid state name, try aliases and Levenshtein distance in case of misspelling.
        @param: name - State name
        @param: levensht_thresh - Threshold for Levenshtein distance
        @return: cononical state name or None
    """
    if name:
        name = name.strip()
    if not name:
        return None
    # Try looking for among state names
    if name in US_STATES:
        return name
    # Normalize and try again
    norm_name = normalize(name)
    if norm_name in US_STATES:
        return norm_name
    # Try among aliases
    state = STATES_ALIASES.get(norm_name, None)
    if state:
        return state
    # And final attempt with Levenshtein metric
    max_ratio, closest_name = find_closest_name(norm_name, list(STATES_ALIASES.keys()))
    if max_ratio >= levensht_thresh:
        return closest_name
    return None
