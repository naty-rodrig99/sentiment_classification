

def remove_plurals(word):
    if word.endswith('s'):
        return word[:-1] 
    return word

def preprocess(text):
    # Remove HTML tags
    processed_text = ''
    inside_tag = False
    for char in text:
        if char == '<':
            inside_tag = True
        elif char == '>':
            inside_tag = False
        elif not inside_tag:
            processed_text += char
    
    # Remove @, #, http, and www
    processed_text = processed_text.replace('@', '').replace('#', '')
    if 'http' in processed_text:
        processed_text = processed_text.split('http')[0]
    if 'www.' in processed_text:
        processed_text = processed_text.split('www.')[0]

    # Convert to lowercase
    processed_text = processed_text.lower()

    # Remove punctuation and non-alphabetic characters (emojis) - keep only letters and spaces
    processed_text = ''.join([char if char.isalpha() or char == ' ' else '' for char in processed_text])

    # Split into individual words
    tokens = processed_text.split()

    # Lemmatization: Convert plurals to singular
    #tokens = [remove_plurals(token) for token in tokens]

    # Join into a cleaned string
    return ' '.join(tokens)