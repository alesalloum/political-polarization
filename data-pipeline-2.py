import os
import gzip
import json
import re
import logging
import pickle

import pandas as pd
from tqdm import tqdm

from libvoikko import Voikko
import networkx as nx


v = Voikko(language="fi", path="./dict")

# Helper functions

def check_retweet_status(obj):
    """
    Checks whether an object is a retweet.

    Args:
        obj (dict): A dictionary representing the object to check.

    Returns:
        bool: True if the object is a retweet, False otherwise.
    """
    return "referenced_tweets" in obj and obj["referenced_tweets"][0]["type"] == "retweeted"

def extract_tokens(text):
    """
    Extracts all words from a text string.

    Args:
        text (str): The text string to extract words from.

    Returns:
        list: A list of all words in the text string, in the order they appear.
    """
    return re.findall(r"\b[A-Za-z-äö]+\b", text)

def remove_short_tokens(text, minimum_length = 2):
    """
    Removes all words from a list that are shorter than a given minimum length.

    Args:
        text (list): The list of words to remove short words from.
        minimum_length (int, optional): The minimum length of words to keep. Defaults to 2.

    Returns:
        list: A list of all words in the input list that have a length greater than or equal to minimum_length.
    """
    return [w for w in text if len(w) >= minimum_length]

def lemmatize_tokens(tokens):
    """
    Lemmatizes a list of tokens using the Voikko lemmatizer.

    Args:
        tokens (list): A list of tokenized words to be lemmatized.

    Returns:
        list: A list of the lemmatized tokens.

    Raises:
        ImportError: If the `libvoikko` package is not installed.

    Note:
        This function requires the `libvoikko` package to be installed. The lemmatization process
        uses the Finnish Voikko lemmatizer, which can analyze and lemmatize Finnish words.
        If a token cannot be analyzed, the original token is added to the output list.
    """
    lemmatized_tokens = []
    for token in tokens:
        lem_analysis = v.analyze(token)
        if not lem_analysis:
            lemmatized_tokens.append(token)
        else:
            lemma = lem_analysis[0].get("BASEFORM")
            lemmatized_tokens.append(lemma)

    return lemmatized_tokens

def lowercase_tokens(tokens):
    """
    Converts all strings in the given list to lowercase.

    Parameters:
    strings (list): A list of strings.

    Returns:
    list: A new list with all the strings converted to lowercase.
    """
    return [s.lower() for s in tokens]

def check_relevancy(tokens, selected_keywords):
    """
    Determines whether a set of tokens contains any of the selected keywords.
    
    Args:
        tokens (set): A set of string tokens to check for relevancy.
        selected_keywords (set): A set of string keywords to match against the tokens.
    
    Returns:
        bool: True if the intersection of the token set and the selected keyword set is non-empty,
        indicating that at least one keyword was found in the token set. False otherwise.
    """
    return bool(tokens & selected_keywords)

def preprocess_keywords(keywords):

    lemmatized_keywords = lowercase_tokens(
                            lemmatize_tokens(keywords)
                        )

    return set(lemmatized_keywords)

def remove_duplicates(raw_data):
    """
    Remove duplicate objects from a list of JSON objects.

    Args:
        raw_data (list): A list of JSON objects, each represented as a dictionary.

    Returns:
        list: A new list of JSON objects with duplicates removed.
    """

    print(f"Number of json objects found is {len(raw_data)}.")
    
    no_duplicates = []

    for obj in tqdm(raw_data):
        if obj not in no_duplicates:
            no_duplicates.append(obj)

    print(f"After removing the duplicates, we are left with {len(no_duplicates)} objects.")

    return no_duplicates

# Process functions

def load_data(input_data_dir):

    twitter_files = sorted(os.listdir(input_data_dir))[:-1]
    raw_data = []

    for tweets_file in tqdm(twitter_files):
        #logging.info(f"Loading file {tweets_file} now.")
        with gzip.open(filename = os.path.join(input_data_dir, tweets_file), mode = 'rb') as f_tweets:
            for line in f_tweets:
                tweet = json.loads(line)
                raw_data.append(tweet)
    # add pickling later
    return raw_data

def filter_data(data):

    data_preprocessed_1 = remove_duplicates(data)
    data_preprocessed_2 = [obj for obj in data_preprocessed_1 if check_retweet_status(obj)]

    return data_preprocessed_2

def preprocess_text(text):
    """
    Preprocesses the input text by applying a series of text cleaning and normalization techniques,
    including tokenization, lowercasing, lemmatization, and removal of short tokens.

    Args:
    - text (str): The input text to preprocess.

    Returns:
    - processed_text (list of str): A list of preprocessed tokens, in lowercase and lemmatized form,
      with short tokens (i.e., tokens with length <= 2) removed.

    Example:
        text = "The quick brown fox jumps over the lazy dog."
        preprocess_text(text)
    ['quick', 'brown', 'fox', 'jump', 'over', 'lazy', 'dog']
    """

    processed_text = lowercase_tokens(
                        lemmatize_tokens(
                            remove_short_tokens(
                                extract_tokens(text)
                            )
                        )
                    )
    
    return set(processed_text)

def save_network_data(edge_data, path, network_context):
    """Saves network data to files in the specified directory.

    Args:
        edge_data (list): List of tuples containing edge data.
        path (str): Path to the directory where files will be saved.
        network_context (str): Name of the network to be saved.

    Returns:
        None

    Saves two files in the specified directory:
        - A comma-separated text file containing edge data (source, target, timestamp),
          named <network_context>_edgelist.txt
        - A GraphML file containing the network graph, named <network_context>_net.graphml
    """
    
    df = pd.DataFrame(edge_data, columns=["source", "target", "timestamp"])
   
    full_path_edgelist = os.path.join(path, network_context + "_edgelist.txt")
    df.to_csv(full_path_edgelist, index=False, header=False)

    full_path_graphml = os.path.join(path, network_context + "_net.graphml")
    G = nx.from_pandas_edgelist(df, create_using = nx.DiGraph())
    nx.write_graphml_lxml(G, full_path_graphml)

# MAIN

# 2019 PARLIAMENT ELECTIONS

KEYWORDS_2019 = ["ajatuspoliisi","aktiivimalli", "amkopiskelijat","amkopiskelijatutkimus","ammatillinenkoulutus","arvopohja","arvot","asunnottomuus","avoimuus","demarit","demokratia","digitalisaatio","eduskunta","eduskuntavaaliehdokas","eduskuntavaalit","eduskuntavaalit2019","ehdokas","eivihapuheelle","ekvaalit","ekvaalit2019","eläke","enemmänkuinlukio","ennakkoäänestys","ennaltaehkäisy","epvaalit","eriarvoisuus","esiopetus","esteettömyys","eupuheenjohtajuus","euroopanunioni","europarlamenttivaalit","eurovaalit","eurovaalit2019","euvaalit2019","euvostoliitto","faktantarkistus","fiilisvaalit","hallitus","hallitusohjelma","hiilinielu","hiilivarasto","huoltosuhde","huumeet","hybridiuhat","hyvinvointi","hyvinvointiasuomeen","hyvinvointivaltio","hyväveli","ihmisarvo","ihmisoikeudet","ilmasto","ilmastolakko","ilmastonmuutos","ilmastopolitiikka","ilmastovaalit","innovaatio","innovaatiot","isvaalitentti","itämeri","itämerivaalit","joseitiedettä","jotainrajaa","journalismi","julkisuus","jytkyäpukkaa","järkivihreä","kaivokset","kaivoslaki","kaivostoiminta","kaivosvero","kampanjastartti","kansalaisaloite","kansanedustaja","kansanvallanmvp","kasvasuomi","kasvu","kdpuolue","kemikalisaatio","keskusta","keskustelu","kestävyysvaje","kieltolaki","kiitossipilä","kirjeäänestys","kokemusasiantuntija","kokoomus","kokoomusristeily","kokoomusristeily2019","kolmikanta","kotouttaminen","koulu","koulutuksestaeileikata","koulutus","koulutuslupaus","koulutusratkaisee","koulutusvaalit","kristillisdemokraatit","kulttuuripolitiikka","kuntavaalit","kvopiskelijat","köyhyys","lainsuojattomat","lapsenoikeudet","lapsenraiskaus","lapset","lapsistrategia","liikenne","liikenyt","luonnonsuojelu","luonto","luotammesuomeen","luottamus","luottamusvaalit2019","maahanmuuttaja","maahanmuutto","maahanmuuttopolitiikka","maahanmuuttovaalit","maahantunkeutuja","maakuntauudistus","mamu","matu","meitäonliianvähän","metsä","mielenterveys","mitkävaalit","monikulttuuri","monikulttuurisuus","monimuotoisuus","moraalikato","muslimi","nato","nuoret","nuortenvaalit","nytonpakko","näytäluontosi","oikealla","oikeudenmukaisuus","oikeuselää","opehommat","opettajanääni","opetus","opintotuki","opiskelijat","oppivelvollisuus","osaajapula","osaaminen","osallisuus","pakkopalautukset","paneelikeskustelu","paperittomat","parempaantyöelämään","parempaapäihdepolitiikkaa","perhe","perheet","perhevapaa","perhevapaat","perhevapaauudistus","perheystävällisyys","persut","perussuomalaiset","perustulo","perustuslaki","pienipalarakkautta","piraatit","piraattipuolue","poliisi","poliitikko","politiikallaonväliä","politiikka","politiikkaradio","politiikkka","populismi","presidentinvaali","presidentinvaalit","propaganda","pshelsinki","puolue","puolueet","puoluekanta","pääministeri","radiovapaahelsinki","raippavero","raiskaukset","raiskaus","rajatkiinni","rasismi","rasistit","refugees","rikkaus","rikollisuus","rikos","rkp","saatehdä","sak","samasuunta","samasuurivaaliopasjokapäivä","sananvapaus","sananvastuu","sdp","seksuaalirikokset","seksuaalivähemmistöt","selkärankaa","seniorit","sensuuri","setämiehet","setämies","siirtolaiset","siksips","siksitiede","sininentulevaisuus","siniset","sinistenehdokkaaksi","sisäilma","sisäilmalupaus","sisäinenturvallisuus","sivistys","soidensuojelu","somali","sosiaaliturva","sote","soteuudistus","stopgrooming","sukupolvipolitiikka","sukupuuttoaalto","suomenkansa","suomi","suomi2019","suomi2030","suomiluottaa","suomionrasistinen","suostumus2018","suunterveys","suvaitsevaisuus","synnytystalkoot","syntyvyys","syrjintä","syrjäytyminen","talous","talouspolitiikka","tasaarvo","teollisuus","teollisuusliitto","terrorismi","terveys","terveysbisnes","tiedevaalit","toimintaedellytykset","tphakijat","translaki","tulejakysy","tulevaisuus","tulevaisuuslinja","turku","turpo","turvallisuus","turvapaikanhakija","turvapaikanhakijat","turvapaikanhakutulva","turvapaikka","turve","tutkimus","työ","työelämä","työllisyys","työllisyyys","työnmurros","työttömyysturva","uutinen","vaalihuijaus","vaalihäirintä","vaalikone","vaalikoneet2020","vaalilupaus","vaalimanipulointi","vaalipäivä","vaalirahoitus","vaalirauha","vaalit","vaalit19","vaalit2019","vaalitentti","vaalivaikuttaminen","vaalivilppi","vaikuttaminen","valehtelu","valta","vammaiset","vapaidenvaltakunta","vapausjavastuu","vasemmisto","vasemmistoliitto","vasvaltuusto","verkkokeskustelu","verotus","vientiteollisuus","viestintä","vihapuhe","vihervasemmisto","vihreät","virkanimitykset","voimavara","yhdenvertaisuus","yhteiskunta","yksityisyyseiolerikos","yliopisto","yliopistot","ympäristö","yrittäjyys","yrittäjyys2019","yrittäjät","äänestys","äänestyskäyttäytyminen","äänestäminen","äänielämälle","äänilapselle","äänityölle","äärioikeisto"]

CLIMATE_2019 = ["hiilinielu", "hiilivarasto", "ilmasto", "ilmastolakko", "ilmastonmuutos", "ilmastopolitiikka", "ilmastovaalit", "luonnonsuojelu", "monimuotoisuus", "nytonpakko", "selkärankaa", "soidensuojelu", "sukupuuttoaalto", "turve", "ympäristö"]
IMMIGRATION_2019 = ["jotainrajaa", "kotouttaminen", "lapsenraiskaus", "maahanmuuttaja", "maahanmuutto", "maahanmuuttopolitiikka", "maahanmuuttovaalit", "maahantunkeutuja", "mamu", "matu", "monikulttuuri", "monikulttuurisuus", "muslimi", "pakkopalautukset", "paperittomat", "rajatkiinni", "rasismi", "rasistit", "refugees", "seksuaalirikokset", "siirtolaiset", "stopgrooming", "suomionrasistinen", "suvaitsevaisuus", "tphakijat", "turvapaikanhakija", "turvapaikanhakijat", "turvapaikanhakutulva", "turvapaikka"]
ECONOMIC_POLICY_2019 = ["talous", "talouspolitiikka", "verotus", "vientiteollisuus", "työllisyys", "työllisyyys", "kestävyysvaje", "huoltosuhde"]
SOCIAL_SECURITY_2019 = ["aktiivimalli", "asunnottomuus", "eläke", "eriarvoisuus", "hyvinvointivaltio", "köyhyys", "perustulo", "sosiaaliturva", "sote", "soteuudistus", "terveysbisnes", "työttömyysturva", "perhevapaa", "perhevapaat", "perhevapaauudistus"]
EDUCATION_2019 = ["amkopiskelijat", "amkopiskelijatutkimus", "ammatillinenkoulutus", "enemmänkuinlukio", "koulutuksestaeileikata", "koulutus", "koulutuslupaus", "koulutusratkaisee", "koulutusvaalit", "kvopiskelijat", "opintotuki", "siksitiede", "sivistys", "joseitiedettä", "siksitiede", "tiedevaalit", "tutkimus", "yliopisto", "yliopistot"]

SDP_2019 = ["SDP", "demarit", "samasuunta", "sdp", "tulevaisuuslinja"]
FINNS_2019 = ["jytkyäpukkaa", "persut", "perussuomalaiset", "pshelsinki", "siksips"]
NATIONAL_2019 = ["kokoomus", "kokoomusristeily", "kokoomusristeily2019", "luotammesuomeen", "oikealla"]
CENTER_2019 = ["keskusta"]
GREEN_2019 = ["järkivihreä", "näytäluontosi", "vihreät"]
LEFT_2019 = ["vasemmisto", "vasemmistoliitto", "vasvaltuusto"]
PARTIES_2019 = ["äärioikeisto", "demarit", "järkivihreä", "jytkyäpukkaa", "kdpuolue", "keskusta", "kokoomus", "kokoomusristeily", "kokoomusristeily2019", "kristillisdemokraatit", "liikenyt", "luotammesuomeen", "näytäluontosi", "oikealla", "persut", "perussuomalaiset", "piraatit", "piraattipuolue", "pshelsinki", "rkp", "samasuunta", "sdp", "siksips", "sininentulevaisuus", "siniset", "sinistenehdokkaaksi", "tulevaisuuslinja", "vasemmisto", "vasemmistoliitto", "vasvaltuusto", "vihervasemmisto", "vihreät"]

# 2023 PARLIAMENT ELECTIONS

KEYWORDS_2023 = ["ajatuspoliisi","aktiivimalli","ansiosidonnainen","arvopohja","arvot","asumistuki","asuminen","asunnottomuus","avoimuus","demarit","demokratia","digitalisaatio","eduskunta","eduskuntavaaliehdokas","eduskuntavaalit","eduskuntavaalit2023","ehdokas","eivihapuheelle","eläke","“energian hinta”","ennakkoäänestys","ennaltaehkäisy","eriarvoisuus","esiopetus","esteettömyys","eteenpäin","eupuheenjohtajuus","euroopanunioni","“euroopan unioni”","EU","euvostoliitto","feminismi","hallitus","hallituskriisi","hallitusohjelma","hiilineutraali","hiilinielu","hiilivarasto","huoltosuhde","hybridiuhat","hyvinvointi","hyvinvointiasuomeen","hyvinvointivaltio","hyväveli","ihmisarvo","ihmisoikeudet","ilmasto","ilmastokriisi","ilmastolakko","ilmastonmuutos","ilmastohuijaus","ilmastopolitiikka","ilmastovaalit","isvaalitentti","jengit","jotainrajaa","journalismi","julkisuus","jytkyäpukkaa","järjenääni","järkivihreä","kaivokset","kaivostoiminta","kaivosvero","kampanjastartti","kansalaisaloite","kansanedustaja","kansanvallanmvp","kasvu","kaupunkisuunnittelu","kdpuolue","kehysriihi","kemikalisaatio","kepu","keskusta","keskustasekotimainen","keskustelu","kestävyysvaje","kestäväkehitys","kestävämpihelsinki","kieltolaki","kiky","kirjeäänestys","kokemusasiantuntija","kokoomus","kokoomusristeily","kolmikanta","korjausliike","korona","kotouttaminen","koulu","koulutuksestaeileikata","koulutus","koulutuslupaus","koulutusratkaisee","kristillisdemokraatit","kulttuuri","kulttuuriala","kulttuuripolitiikka","kvopiskelijat","köyhyys","lapsenoikeudet","lapsenraiskaus","lapset","lapsistrategia","lastensuojelu","leikattavaalöytyy","liberaali puolue","liikenne","liikenyt","luonnonsuojelu","luonto","luontokato","luotammesuomeen","luottamus","maahanmuuttaja","maahanmuutto","maahanmuuttopolitiikka","maahanmuuttovaalit","maahantunkeutuja","mamu","mamukiima","matu","metsä","metsät","mielenterveys","monikulttuuri","monikulttuurisuus","monimuotoisuus","moraalikato","muslimi","nato","neuvostoliitto","nuoret","nuorisovaalit","nuortenvaalit","nytonpakko","oikeaaika","oikealla","oikeudenmukainensiirtymä","oikeudenmukaisuus","oikeusministeriö","opehommat","opettajanääni","opetus","opintotuki","opiskelijat","oppivelvollisuus","osaajapula","osaaminen","osallisuus","pakkopalautukset","paneelikeskustelu","paperittomat","parempaantyöelämään","parempaapäihdepolitiikkaa","perhe","perheet","perhevapaa","perhevapaat","perhevapaauudistus","perheystävällisyys","persut","perussuomalaiset","perustulo","perustuslaki","piraatit","piraattipuolue","poliisi","poliitikko","politiikallaonväliä","politiikka","politiikkaradio","politiikkka","populismi","propaganda","ps2023","pshelsinki","puheenjohtajatentti","puoliväliriihi","puolue","puolueet","puoluekanta","pyörävaalit","pääministeri","radiovapaahelsinki","rahaaon","raippavero","raiskaukset","raiskaus","rajatkiinni","rasismi","rasistit","refugees","rikkaus","rikollisuus","rikos","rkp","rokotteet","rokotukset","rokotus","saatehdä","sak","samasuunta","samasuurivaaliopasjokapäivä","sananvapaus","sananvastuu","sdp","seksuaalirikokset","seksuaalivähemmistöt","selkärankaa","seniorit","sensuuri","setämiehet","setämies","siirtolaiset","siivouspäivä","siksips","siksitiede","sinivalkoinensiirtymä","“sinivalkoinen siirtymä”","sinäpäätät","sisäinenturvallisuus","sivistys","soidensuojelu","somali","s2-oppilas","sosiaaliturva","sote","soteuudistus","stopgrooming","sukupolvipolitiikka","sukupuuttoaalto","suomenkansa","suomi","suomi2023","suomi2030","suomiluottaa","suomionrasistinen","suomitakaisin","suvaitsevaisuus","synnytystalkoot","syntyvyys","syrjintä","säästöt","säästäminen","syrjäytyminen","talous","talousjailmasto","talouspolitiikka","tasaarvo","taustaltanäkyväksi","teollisuus","teollisuusliitto","terrorismi","terveys","terveysbisnes","tiede","tiedevaalit","toimintaedellytykset","tphakijat","translaki","tulejakysy","tulevaisuus","tulevaisuuslinja","turpo","turvallisuus","turvapaikanhakija","turvapaikanhakijat","turvapaikanhakutulva","turvapaikka","turve","tutkimus","työllisyys","työllisyyys","työttömyysturva","uutinen","vaalihuijaus","vaalihäirintä","vaalikone","vaalikoneet2023","vaalilupaus","vaalimanipulointi","vaalipäivä","vaalirahoitus","vaalirauha","vaalit","vaalit2023","vaalit23","vaalitentti","vaalivaikuttaminen","vaalivilppi","vaikuttaminen","valehtelu","valta","valtionvelka","vanhukset","vapaidenvaltakunta","“vapaus valita”","vapausjavastuu","vasemmisto","vasemmistoliitto","vasvaltuusto","velka","verkkokeskustelu","verotus","vientiteollisuus","viestintä","vihapuhe","vihervasemmisto","vihreät","virkanimitykset","voimavara","väestönvaihdos","väestönvaihto","yhdenvertaisuus","yhteiskunta","yksityisyyseiolerikos","yliopisto","yliopistot","ympäristö","ympäristövaalit","yrittäjyys","yrittäjyys2023","yrittäjät","äänestys","äänestyskäyttäytyminen","äänestäminen","äänielämälle","äänilapselle","ääniluonnolle","äänityölle","äärioikeisto"]
NONUNIVERSAL_KEYWORDS_2023 = ["ajatuspoliisi", "aktiivimalli", "ansiosidonnainen", "arvopohja", "arvot", "asumistuki", "asuminen", "asunnottomuus", "avoimuus", "demarit", "demokratia", "digitalisaatio", "eduskunta", "eduskuntavaaliehdokas", "eduskuntavaalit", "eduskuntavaalit2023", "ehdokas", "eivihapuheelle", "eläke", "“energian hinta”", "ennakkoäänestys", "ennaltaehkäisy", "eriarvoisuus", "esiopetus", "esteettömyys", "eteenpäin", "eupuheenjohtajuus", "euroopanunioni", "“euroopan unioni”", "euvostoliitto", "feminismi", "hallitus", "hallituskriisi", "hallitusohjelma", "hiilineutraali", "hiilinielu", "hiilivarasto", "huoltosuhde", "hybridiuhat", "hyvinvointi", "hyvinvointiasuomeen", "hyvinvointivaltio", "hyväveli", "ihmisarvo", "ihmisoikeudet", "ilmasto", "ilmastokriisi", "ilmastolakko", "ilmastonmuutos", "ilmastohuijaus", "ilmastopolitiikka", "ilmastovaalit", "isvaalitentti", "jengit", "jotainrajaa", "journalismi", "julkisuus", "jytkyäpukkaa", "järjenääni", "järkivihreä", "kaivokset", "kaivostoiminta", "kaivosvero", "kampanjastartti", "kansalaisaloite", "kansanedustaja", "kansanvallanmvp", "kasvu", "kaupunkisuunnittelu", "kdpuolue", "kehysriihi", "kemikalisaatio", "kepu", "keskusta", "keskustasekotimainen", "keskustelu", "kestävyysvaje", "kestäväkehitys", "kestävämpihelsinki", "kieltolaki", "kiky", "kirjeäänestys", "kokemusasiantuntija", "kokoomus", "kokoomusristeily", "kolmikanta", "korjausliike", "kotouttaminen", "koulu", "koulutuksestaeileikata", "koulutus", "koulutuslupaus", "koulutusratkaisee", "kristillisdemokraatit", "kulttuuri", "kulttuuriala", "kulttuuripolitiikka", "kvopiskelijat", "köyhyys", "lapsenoikeudet", "lapsenraiskaus", "lapset", "lapsistrategia", "lastensuojelu", "leikattavaalöytyy", "liberaali puolue", "liikenne", "liikenyt", "luonnonsuojelu", "luonto", "luontokato", "luotammesuomeen", "luottamus", "maahanmuuttaja", "maahanmuutto", "maahanmuuttopolitiikka", "maahanmuuttovaalit", "maahantunkeutuja", "mamu", "mamukiima", "matu", "metsä", "metsät", "mielenterveys", "monikulttuuri", "monikulttuurisuus", "monimuotoisuus", "moraalikato", "muslimi", "neuvostoliitto", "nuoret", "nuorisovaalit", "nuortenvaalit", "nytonpakko", "oikeaaika", "oikealla", "oikeudenmukainensiirtymä", "oikeudenmukaisuus", "oikeusministeriö", "opehommat", "opettajanääni", "opetus", "opintotuki", "opiskelijat", "oppivelvollisuus", "osaajapula", "osaaminen", "osallisuus", "pakkopalautukset", "paneelikeskustelu", "paperittomat", "parempaantyöelämään", "parempaapäihdepolitiikkaa", "perhe", "perheet", "perhevapaa", "perhevapaat", "perhevapaauudistus", "perheystävällisyys", "persut", "perussuomalaiset", "perustulo", "perustuslaki", "piraatit", "piraattipuolue", "poliisi", "poliitikko", "politiikallaonväliä", "politiikka", "politiikkaradio", "politiikkka", "populismi", "ps2023", "pshelsinki", "puheenjohtajatentti", "puoliväliriihi", "puolue", "puolueet", "puoluekanta", "pyörävaalit", "pääministeri", "radiovapaahelsinki", "rahaaon", "raippavero", "raiskaukset", "raiskaus", "rajatkiinni", "rasismi", "rasistit", "rikkaus", "rikollisuus", "rikos", "rkp", "rokotteet", "rokotukset", "rokotus", "saatehdä", "sak", "samasuunta", "samasuurivaaliopasjokapäivä", "sananvapaus", "sananvastuu", "seksuaalirikokset", "seksuaalivähemmistöt", "selkärankaa", "seniorit", "sensuuri", "setämiehet", "setämies", "siirtolaiset", "siivouspäivä", "siksips", "siksitiede", "sinivalkoinensiirtymä", "“sinivalkoinen siirtymä”", "sinäpäätät", "sisäinenturvallisuus", "sivistys", "soidensuojelu", "s2-oppilas", "sosiaaliturva", "sote", "soteuudistus", "sukupolvipolitiikka", "sukupuuttoaalto", "suomenkansa", "suomi", "suomi2023", "suomi2030", "suomiluottaa", "suomionrasistinen", "suomitakaisin", "suvaitsevaisuus", "synnytystalkoot", "syntyvyys", "syrjintä", "säästöt", "säästäminen", "syrjäytyminen", "talous", "talousjailmasto", "talouspolitiikka", "tasaarvo", "taustaltanäkyväksi", "teollisuus", "teollisuusliitto", "terrorismi", "terveys", "terveysbisnes", "tiede", "tiedevaalit", "toimintaedellytykset", "tphakijat", "translaki", "tulejakysy", "tulevaisuus", "tulevaisuuslinja", "turpo", "turvallisuus", "turvapaikanhakija", "turvapaikanhakijat", "turvapaikanhakutulva", "turvapaikka", "turve", "tutkimus", "työllisyys", "työllisyyys", "työttömyysturva", "uutinen", "vaalihuijaus", "vaalihäirintä", "vaalikone", "vaalikoneet2023", "vaalilupaus", "vaalimanipulointi", "vaalipäivä", "vaalirahoitus", "vaalirauha", "vaalit", "vaalit2023", "vaalit23", "vaalitentti", "vaalivaikuttaminen", "vaalivilppi", "vaikuttaminen", "valehtelu", "valta", "valtionvelka", "vanhukset", "vapaidenvaltakunta", "“vapaus valita”", "vapausjavastuu", "vasemmisto", "vasemmistoliitto", "vasvaltuusto", "velka", "verkkokeskustelu", "verotus", "vientiteollisuus", "viestintä", "vihapuhe", "vihervasemmisto", "vihreät", "virkanimitykset", "voimavara", "väestönvaihdos", "väestönvaihto", "yhdenvertaisuus", "yhteiskunta", "yksityisyyseiolerikos", "yliopisto", "yliopistot", "ympäristö", "ympäristövaalit", "yrittäjyys", "yrittäjyys2023", "yrittäjät", "äänestys", "äänestyskäyttäytyminen", "äänestäminen", "äänielämälle", "äänilapselle", "ääniluonnolle", "äänityölle", "äärioikeisto"]

CLIMATE_2023 = ["hiilineutraali", "ilmastohuijaus", "ilmastokriisi", "järkivihreä", "kaivokset", "kaivostoiminta", "kaivosvero", "kaupunkisuunnittelu", "kemikalisaatio", "liikenne", "luonto", "luontokato", "metsä", "metsät", "sinivalkoinensiirtymä", "talousjailmasto", "teollisuus", "teollisuusliitto", "ympäristövaalit", "ääniluonnolle", "“sinivalkoinen siirtymä”"] + CLIMATE_2019
IMMIGRATION_2023 = ["eivihapuheelle", "eriarvoisuus", "ihmisarvo", "ihmisoikeudet", "jengit", "kotouttaminen", "mamukiima", "osaajapula", "raiskaukset", "raiskaus", "rikkaus", "rikollisuus", "rikos", "s2-oppilas", "sananvapaus", "sananvastuu", "somali", "suomenkansa", "suomitakaisin", "syntyvyys", "terrorismi", "työllisyys", "vihapuhe", "väestönvaihdos", "väestönvaihto", "yhdenvertaisuus", "äärioikeisto"] + IMMIGRATION_2019
ECONOMIC_POLICY_2023 = ["aktiivimalli", "ansiosidonnainen", "kasvu", "kiky", "kolmikanta", "osaajapula", "osaaminen", "perustulo", "sak", "säästäminen", "säästöt", "talousjailmasto", "teollisuus", "teollisuusliitto", "työttömyysturva", "tutkimus", "valtionvelka", "velka", "yrittäjät", "yrittäjyys2023", "yrittäjyys", "äänityölle"] + ECONOMIC_POLICY_2019

# :------------------: #

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define input and output directories and network name
input_data_dir = "../keywords_non_universal_stream"
output_data_dir = "./keywords_non_universal_stream_processed"
network_context = "ECONOMIC_POLICY"

def run_pipeline():

    logging.info(f"Starting data pipeline for {network_context}...")

    # Load data from input directory
    data = load_data(input_data_dir)
    logging.info("Data loaded successfully")

    data = filter_data(data)
    logging.info("Data filtered successfully")

    KEYWORDS_SELECTED = preprocess_keywords(network_context + "_2023")
    EDGE_DATA = []

    for retweet in data:

        tweet_text = retweet["referenced_tweets"][0]["tweet"]["text"]
    
        TEXT_TOKENS = preprocess_text(tweet_text)

        # CHECKPOINT OPERATION STARTS
        # Define the filename to save the pickle
        filename = './checkpoint/processed_tokens.pkl'

        # Open the file in binary mode and write the pickled object to it
        with open(filename, 'wb') as f:
            pickle.dump(TEXT_TOKENS, f)
        logging.info("Checkpoint saved as a pickle")
        # CHECKPOINT OPERATION ENDS
        
        if check_relevancy(TEXT_TOKENS, KEYWORDS_SELECTED):
        
            retweeter_node = retweet["author_id"]
            retweeted_node = retweet["referenced_tweets"][0]["tweet"]["author_id"]
            timestamp = retweet["created_at"]

            EDGE_DATA.append((retweeter_node, retweeted_node, timestamp))
        
    logging.info("Edge formation done successfully")

    save_network_data(EDGE_DATA, output_data_dir, network_context)
    logging.info(f"Network {network_context} data saved successfully")

if __name__ == "__main__":
    run_pipeline()