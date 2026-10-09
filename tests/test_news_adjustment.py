from rag.adjust import classify


def _chunk(text):
    return [{"text": text, "url": "https://example.com/report"}]


def test_substrings_do_not_invent_an_injury_or_suspension():
    assert classify(_chunk("Cole Palmer talks about urban football culture."), "cole palmer") is None
    assert classify(_chunk("Elversberg are a miracle village."), "sepp van den berg") is None


def test_negated_doubt_does_not_reduce_a_prediction():
    assert classify(_chunk("There is no doubt Marcus Rashford can play."), "marcus rashford") is None


def test_another_persons_news_does_not_flag_the_player():
    assert classify(_chunk("Scott McTominay is out for two weeks."), "alex scott") is None
    assert classify(_chunk("Ferran Torres is ruled out."), "pau torres") is None


def test_explicit_player_availability_is_detected():
    assert classify(_chunk("Bukayo Saka is ruled out for Saturday."), "bukayo saka")[0] == "OUT"
    assert classify(_chunk("Bukayo Saka has a knock and faces a late fitness test."), "bukayo saka")[0] == "DOUBT"
    assert classify(_chunk("Bukayo Saka is back in training."), "bukayo saka")[0] == "BOOST"


def test_a_different_sentence_does_not_supply_the_injury_signal():
    assert classify(_chunk("Bukayo Saka scored. Martin Odegaard is ruled out."), "bukayo saka") is None


def test_another_players_clause_does_not_supply_the_injury_signal():
    assert classify(_chunk("Bukayo Saka scored, but Martin Odegaard is ruled out."), "bukayo saka") is None

def test_auxiliary_negation_does_not_invent_unavailability():
    assert classify(_chunk("Bukayo Saka has not been ruled out."), "bukayo saka") is None
    assert classify(_chunk("Bukayo Saka is not currently suspended."), "bukayo saka") is None


def test_coordinated_subject_does_not_supply_another_players_signal():
    assert classify(_chunk("Cole Palmer is available and Reece James is ruled out."), "cole palmer") is None
    assert classify(_chunk("Bukayo Saka trained, Martin Odegaard is ruled out."), "bukayo saka") is None
