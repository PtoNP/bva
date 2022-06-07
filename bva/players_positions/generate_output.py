import pandas as pd
from players_positions.court_context import CourtContext

def generate_hitmap(players_positions_path, hits_df, stroke_df):

    bcc = CourtContext()
    players_positions = pd.read_csv(players_positions_path)

    counter = 0
    while True:
        players_visible = players_positions.loc[counter]['player_A_visible']

        if players_visible:
            pA_x = players_positions.loc[counter]['player_A_court_x']
            pA_y = players_positions.loc[counter]['player_A_court_y']
            pB_x = players_positions.loc[counter]['player_B_court_x']
            pB_y = players_positions.loc[counter]['player_B_court_y']

            # remove a bit (75cm) of y for B, the front foot is upper
            distance_A_to_T = ((pA_x - bcc.up_service_middle[1])**2 \
                             + (pA_y - 75 - bcc.up_service_middle[0])**2 \
                              )**.5

            distance_B_to_T = ((pB_x - bcc.bottom_service_middle[1])**2 \
                             + (pB_y - bcc.bottom_service_middle[0])**2 \
                              )**.5

            if distance_A_to_T < distance_B_to_T:
                server = 1
            else:
                server = -1

            break
        counter += 1

    images = []
    history = { 'player_A' : [], 'player_B': []}
    counter = 0
    player = server
    for hit in hits_df.iterrows():
        if player == 1:
            player_name = 'player_A'
        else:
            player_name = 'player_B'

        if hit[1]['hit'] == 1:
            position_x = players_positions.loc[counter][f'{player_name}_court_x']
            position_y = players_positions.loc[counter][f'{player_name}_court_y']
            attack = stroke_df.loc[counter]['0']
            defense = stroke_df.loc[counter]['1']
            if defense > attack:
                stroke_color = (0,0,255)
            else:
                stroke_color = (255,0,0)

            history[player_name].append([(position_x, position_y),stroke_color])
            player = -player

        img = bcc.drawCourt()

        for h in history['player_A']:
            img = bcc.drawCourtPosition(img, h[0], color=h[1],filled=True)

        for h in history['player_B']:
            img = bcc.drawCourtPosition(img, h[0], color=h[1],filled=True)

        pA_x = players_positions.loc[counter]['player_A_court_x']
        pA_y = players_positions.loc[counter]['player_A_court_y']
        pB_x = players_positions.loc[counter]['player_B_court_x']
        pB_y = players_positions.loc[counter]['player_B_court_y']

        if counter < len(hits_df) -1:
            img = bcc.drawCourtPosition(img, (pA_x,pA_y), color=(0,165,255), radius=20)
            img = bcc.drawCourtPosition(img, (pB_x,pB_y), color=(0,165,255), radius=20)

        images.append(img)
        counter += 1

    return images
