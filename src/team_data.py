# src/team_data.py

TEAM_ABBREVIATIONS = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
}

TEAM_NAMES = {v: k for k, v in TEAM_ABBREVIATIONS.items()}

# NEW: Official team colors for UI enhancements
TEAM_COLORS = {
    'ARI': ('#97233F', '#000000'), 'ATL': ('#A71930', '#000000'), 'BAL': ('#241773', '#000000'),
    'BUF': ('#00338D', '#C60C30'), 'CAR': ('#0085CA', '#101820'), 'CHI': ('#0B162A', '#E64100'),
    'CIN': ('#FB4F14', '#000000'), 'CLE': ('#311D00', '#FF3C00'), 'DAL': ('#041E42', '#869397'),
    'DEN': ('#FB4F14', '#002244'), 'DET': ('#0076B6', '#B0B7BC'), 'GB': ('#203731', '#FFB612'),
    'HOU': ('#03202F', '#A71930'), 'IND': ('#002C5F', '#A2AAAD'), 'JAX': ('#006778', '#9F792C'),
    'KC': ('#E31837', '#FFB81C'), 'LV': ('#000000', '#A5ACAF'), 'LAC': ('#0080C6', '#FFC20E'),
    'LA': ('#003594', '#FFA300'), 'MIA': ('#008E97', '#FC4C02'), 'MIN': ('#4F2683', '#FFC62F'),
    'NE': ('#002244', '#C60C30'), 'NO': ('#D3BC8D', '#101820'), 'NYG': ('#0B2265', '#A71930'),
    'NYJ': ('#125740', '#FFFFFF'), 'PHI': ('#004C54', '#A5ACAF'), 'PIT': ('#FFB612', '#101820'),
    'SF': ('#AA0000', '#B3995D'), 'SEA': ('#002244', '#69BE28'), 'TB': ('#D50A0A', '#34302B'),
    'TEN': ('#0C2340', '#4B92DB'), 'WAS': ('#5A1414', '#FFB612')
}