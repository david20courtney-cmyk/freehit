from flask import Flask, request, jsonify
from flask_cors import CORS
import pulp

app = Flask(__name__)
CORS(app)

@app.route('/optimize', methods=['POST', 'OPTIONS'])
@app.route('/optimise', methods=['POST', 'OPTIONS'])
def optimize_team():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json() or {}
    players = data.get('players', [])
    
    if not players:
        return jsonify({"status": "error", "message": "No players provided"}), 400

    # 1. Discover all unique gameweeks present in the dataset
    unique_gws = set()
    for p in players:
        for gw_info in p.get('gameweeks', []):
            unique_gws.add(gw_info.get('gw'))
    
    sorted_gws = sorted(list(unique_gws))
    
    if not sorted_gws:
        return jsonify({"status": "error", "message": "No gameweek data provided"}), 400

    # Helper Extraction Functions
    def get_price(p):
        val = p.get('price') if p.get('price') is not None else p.get('cost', p.get('now_cost', 0.0))
        try:
            val = float(val)
            # Handle standard FPL prices (e.g. 100 = 10.0m or 85 = 8.5m)
            return val / 10.0 if val > 20.0 else val
        except (ValueError, TypeError):
            return 0.0

    def get_pos(p):
        pos = p.get('position') or p.get('pos') or p.get('element_type') or ''
        pos_str = str(pos).strip().upper()
        
        if pos_str in ['1', 'GK', 'GKP', 'GOALKEEPER']:
            return 'GKP'
        elif pos_str in ['2', 'DEF', 'DEFENDER']:
            return 'DEF'
        elif pos_str in ['3', 'MID', 'MIDFIELDER']:
            return 'MID'
        elif pos_str in ['4', 'FWD', 'FORWARD', 'ATTACKER']:
            return 'FWD'
        return pos_str

    def get_team(p):
        return str(p.get('team') or p.get('team_name') or p.get('club') or '').strip()

    def get_xp(p, target_gw):
        for gw_data in p.get('gameweeks', []):
            if gw_data.get('gw') == target_gw:
                try:
                    return float(gw_data.get('xp', 0.0))
                except (ValueError, TypeError):
                    return 0.0
        return 0.0

    # Dictionaries to store results per GW
    points_per_gameweek = {}
    gameweek_lineups = {}

    # 2. Run optimization for EACH gameweek independently
    for gw in sorted_gws:
        prob = pulp.LpProblem(f"FPL_GW_{gw}_Optimizer", pulp.LpMaximize)
        
        x = {p['id']: pulp.LpVariable(f"x_{p['id']}", cat='Binary') for p in players}
        y = {p['id']: pulp.LpVariable(f"y_{p['id']}", cat='Binary') for p in players}
        
        # OBJECTIVE for this specific Gameweek
        prob += pulp.lpSum(get_xp(p, gw) * x[p['id']] + get_xp(p, gw) * y[p['id']] for p in players)
        
        # CONSTRAINTS
        def_vars = [x[p['id']] for p in players if get_pos(p) == 'DEF']
        
        # 2.1 Dynamic Budget Constraint based on number of defenders
        # Math trick: Cost + 0.5 * DEF <= 85.0
        # If DEF=3 -> Cost <= 83.5, If DEF=4 -> Cost <= 83.0, If DEF=5 -> Cost <= 82.5
        prob += pulp.lpSum(get_price(p) * x[p['id']] for p in players) + 0.5 * pulp.lpSum(def_vars) <= 85.0
        
        # 2.2 Total starting XI size = 11
        prob += pulp.lpSum(x[p['id']] for p in players) == 11
        
        # 2.3 Positional limits
        prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'GKP') == 1
        prob += pulp.lpSum(def_vars) >= 3
        prob += pulp.lpSum(def_vars) <= 5
        prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'MID') >= 2
        prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'MID') <= 5
        prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'FWD') >= 1
        prob += pulp.lpSum(x[p['id']] for p in players if get_pos(p) == 'FWD') <= 3

        # 2.4 Max 3 players per team
        teams = set(get_team(p) for p in players if get_team(p))
        for team in teams:
            if team:
                prob += pulp.lpSum(x[p['id']] for p in players if get_team(p) == team) <= 3

        # 2.5 Captain constraints
        prob += pulp.lpSum(y[p['id']] for p in players) == 1
        for p in players:
            prob += y[p['id']] <= x[p['id']]
            
        # SOLVE
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if status != 1:
            points_per_gameweek[gw] = 0.0
            gameweek_lineups[gw] = {"status": "error", "message": "Infeasible model"}
            continue

        # EXTRACT RESULTS FOR THIS GW
        starting_xi = []
        for p in players:
            if x[p['id']].varValue and x[p['id']].varValue > 0.5:
                # Include the specific gameweek's XP and captaincy status in the player dict
                p_copy = p.copy()
                p_copy['gw_xp'] = get_xp(p, gw)
                p_copy['is_captain'] = bool(y[p['id']].varValue and y[p['id']].varValue > 0.5)
                starting_xi.append(p_copy)
                
        captain = next((p for p in starting_xi if p['is_captain']), None)
        
        total_cost = sum(get_price(p) for p in starting_xi)
        total_xp = pulp.value(prob.objective)

        points_per_gameweek[gw] = round(total_xp, 2)
        gameweek_lineups[gw] = {
            "status": "success",
            "captain_id": captain['id'] if captain else None,
            "total_cost": round(total_cost, 1),
            "total_xp": round(total_xp, 2),
            "starting_xi": starting_xi
        }

    # 3. Return final structured JSON
    return jsonify({
        "status": "success",
        "points_per_gameweek": points_per_gameweek,
        "gameweeks": gameweek_lineups
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
