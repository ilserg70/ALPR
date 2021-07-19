import re
import math

# Alphabet is a set of symbols which is used for licence plates annotation.
# Alphabet consists of:

# 1) Terminal chars
NUMBERS = ['0','1','2','3','4','5','6','7','8','9']
LETTERS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
CHARS = NUMBERS + LETTERS + ['-']

# 2) Meta-symbols of annotation language
OPEN_BR = ['[','(','{','<']
CLOSE_BR = [']',')','}','>']
META = OPEN_BR + CLOSE_BR
#    Where:
#          [] - denotes vertical stacked symbols, examples: [DP], [AMD], ...
#          <> - denotes doubtful symbols in order of confidence, examples: <OQ0>, <EF>, <B8>, <TY>, <XY>, ...
#          () - denotes smaller symbols
#          {} - denotes ???

ALPHABET = CHARS + META

# Symbols that are difficult for people to distinguish. Examples: O-0-Q, B-8, 1-I, S-5, 7-Z-2, M-H-W, F-E, ...  
WARN_CHARS = set(['5','K','1','7','L','8','0','T','F','E','U','D','2','X','Y','9','S','Q','B','I','Z','M','W','O'])

def no_warn(text):
    for c in text:
        if c in WARN_CHARS:
            return False
    return True

def check_annotation(text):
    """ Verification of brackets.
        @param: text - Licence plate text
        @return: True/False
    Examples:
        '[AT]8756'              -> True
        'R[FG(89)]554'          -> True
        '[(AB)(RR)(WS)]7776'    -> True
        '67[<OQ><B8>]YU(state)' -> True
        'DR[<TY]8>'             -> False
        '56<OQ<B8>>OUT'         -> False
        Correct: [..]..[..]..(..); [(..)(..)]; [<..>..<..>]
        Not correct: [..(..]..); <..<..>..>; [..[..]..]; (..(..)..); {..{..}..}; <..[..]..>
    """
    if not text:
        return False
    stack = []
    for c in text:
        if c in "[({<":
            # Before opening a new bracket it is need to close this kind of bracket
            if c in stack:
                return False
            stack.append(c)
        elif c in "])}>":
            # Before closing it is need to open bracket
            if not stack:
                return False
            c_ = stack.pop()
            # Check if last opening bracket is the same kind
            if not (c_=='[' and c==']' or c_=='(' and c==')' or c_=='{' and c=='}' or c_=='<' and c=='>'):
                return False
    # If missed closing bracket(s)
    if stack:
        return False
    return True

def check_chars(text, case_sens=False):
    """ Verification if all symbols are from alphabet.
        @param: text - Licence plate text
        @param: case_sens - case sensitive
        @return: True/False
    Examples:
        '[DP]5690-S' -> True
        '@[DP]5690S' -> False
    """
    if not text:
        return False
    if not case_sens:
        text = text.upper()
    for c in text:
        if c not in ALPHABET:
            return False
    return True

def clean(text):
    """ Text cleaning: 
        to uppercase, remove not alphabet chars, remove empty brackets
        @param: text  - Licence plate text
        @return: text - Purified text
    """
    if text:
        text = text.strip()
    if not check_annotation(text):
        return None
    # To upper and remove all chars not in ALPHABET
    text = ''.join([c for c in text.upper() if c in ALPHABET])
    # Remove all empty brackets: [], (), {}, <>
    text = re.sub(r"(\[\]|\(\)|\{\}|<>)", '', text)
    return text if text else None

def take_first_uncertain(text):
    """ Take the first char from all blocks of uncertainty (angle brackets).
        @param: text  - Licence plate text
        @return: text - Licence plate text with no uncertainties
    Examples:
        <OQ>RT556  -> ORT556
        A<EFP>X888 -> AEX888
        334<8B>W<TY>  -> 3348WT
    """
    if text and '<' in text:
        p = re.compile(r"(<[^<>]+>)")
        for g in p.findall(text):
            text = text.replace(g, g[1])
    return text

def stacked_groups(text):
    """ Extract all stacked groups.
        @param: text  - Licence plate text
        @return: list of stacked groups
    Examples:
        [AB]564[3][DRE]  -> ['AB', 'DRE', '3']
        [(WA)(UTD)(TY)]111 -> ['WA', 'UTD', 'TY']
        {RE}02[675]54(SSS) -> ['675', 'RE', 'SSS']
        [<OQ><EF>]656 -> ['OE']
    """
    text = take_first_uncertain(text)
    
    p0 = re.compile(r"\[([^\[\]]+)\]")
    p1 = re.compile(r"\{([^\{\}]+)\}")
    p2 = re.compile(r"\(([^\(\)]+)\)")
    gg = [g for g in p0.findall(text)] + [g for g in p1.findall(text)] + [g for g in p2.findall(text)]
    groups = set()
    for g2 in gg:
        if '(' in g2:
            for g in p2.findall(g2):
                groups.add(g)
        else:
            groups.add(g2)    
    return list(groups)

def big_chars(text):
    """ Extract big chars and remove all stacked groups.
        @param: text  - Licence plate text
        @return: big text
    Examples:
        [AB]564[3][DRE]  -> 564
        [(WA)(UTD)(TY)]111 -> 111
        {RE}02[675]54(SSS) -> 0254
        [<OQ><EF>]656 -> 656
    """
    text = take_first_uncertain(text)
    
    p0 = re.compile(r"(\[[^\[\]]+\])")
    p1 = re.compile(r"(\{[^\{\}]+\})")
    p2 = re.compile(r"(\([^\(\)]+\))") 
    gg = [g for g in p0.findall(text)] + [g for g in p1.findall(text)] + [g for g in p2.findall(text)]
    for g in gg:
        text = text.replace(g, '')
    return text

def rm_stacked_chars(text):
    """ Remove stacked characters.
        @param: text  - Licence plate text
        @return: text, couple - text with no stacks and list of [..] blocks
    Example:
        '[PD]459870W' -> '459870W'
    """
    stacks = []
    p = re.compile(r"(\[[^\[\]]{1,}\])")
    for g in p.findall(text):
        stacks.append(g[1:-1])
    text = p.sub('', text)
    return text, stacks

def mk_template(text):
    """ 
        [AD]345TYR(Y) -> [##]######(#)
        123TYW -> ######
        88-RTEW -> #######
    """
    return re.sub(r"[\d\w-]", '#', text)

def is_stacked(text):
    if text:
        return sum([c in "[]{}()" for c in text])>0
    return False

def rm_metasymbols(text):
    """ Clear text from meta-symbols
    """
    if text:
        return ''.join([c for c in text if c in CHARS])
    return text

def get_len(text):
    if text:
        return len(rm_metasymbols(text))
    return 0

def get_chars(text):
    """ Get list of chars for plate text. Append underscore for chars in []-groups
    """
    chars = []
    if text:
        is_small = False
        for c in text:
            if c == '[':
                is_small = True
            elif c == ']':
                is_small = False
            elif c in CHARS:
                if is_small:
                    c = c+'_'
                chars.append(c)
    return chars

def add_to_similarity_map(text, sim_map, max_len=12):
    """ Add 'text' statistics to similarity histogram.
    @param: text - Plate text
    @param: sim_map - Similarity histogram
    @param: max_len - Max length of plate text
    @return: sim_map - {
        'small':       [..,POSITION_COUNT,..],
        'big':         [..,POSITION_COUNT,..],
        'numb':        [..,POSITION_COUNT,..],
        'lett':        [..,POSITION_COUNT,..],
        'cnt_lett':    [..,COUNT,..],
        'cnt_numb':    [..,COUNT,..],
        'cnt':         [..,COUNT,..]
        'small_chars': {CHAR:COUNT},
        'big_chars':   {CHAR:COUNT},
    }
    """
    def _add(dest, name, val):
        if name not in dest:
            dest[name] = val
        elif isinstance(val, (int,float)):
            dest[name] += val

    for k in ['small','big','numb','lett']:
        _add(sim_map, k, [0]*max_len)
    for k in ['cnt_lett','cnt_numb','cnt']:
        _add(sim_map, k, [0]*(max_len+1))
    for k in ['small_chars','big_chars']:
        _add(sim_map, k, {})

    cnt = {}
    is_small = False
    pos = 0
    for c in text:
        if c == '[':
            is_small = True
        elif c == ']':
            is_small = False
        elif c in CHARS:
            if pos >= max_len:
                break

            k = 'small' if is_small else 'big'
            _add(sim_map[f"{k}_chars"], c, 1)

            sim_map['small'][pos] += 1 if is_small else 0
            sim_map['big'][pos] += 0 if is_small else 1

            is_numb = c in NUMBERS
            k = 'numb' if is_numb else 'lett'
            _add(cnt, k, 1)
            sim_map['numb'][pos] += 1 if is_numb else 0
            sim_map['lett'][pos] += 0 if is_numb else 1

            pos += 1

    for k, m in cnt.items():
        sim_map[f"cnt_{k}"][m] += 1
    sim_map['cnt'][pos] += 1

def to_similarity_vect(sim_map):
    """ Transform histogram to similarity vector.
    @param: sim_map - Similarity histogram
    @return: sim_vect - [<small>,<big>,<numb>,<lett>,<cnt_lett>,<cnt_numb>,<cnt>,<small_chars>,<big_chars>]
    """
    sim_vect = []
    for k in ['small','big','numb','lett','cnt_lett','cnt_numb','cnt']:
        cnt = max(sum(sim_map[k]), 1)
        sim_vect += [v/cnt for v in sim_map[k]]
    for k in ['small_chars','big_chars']:
        cnt = max(sum(sim_map[k].values()), 1)
        sim_vect += [sim_map[k].get(c, 0)/cnt for c in CHARS]
    return sim_vect

def cosin_similarity(x, y):
    """ Calculation cosin similarity of two vectors.
    """
    xy = sum([x_ * y_ for x_, y_ in zip(x, y)])
    x2 = sum([x_ * x_ for x_ in x])
    y2 = sum([y_ * y_ for y_ in y])
    return xy / (math.sqrt(x2) * math.sqrt(y2)) if x2!=0 and y2!=0 else 0
