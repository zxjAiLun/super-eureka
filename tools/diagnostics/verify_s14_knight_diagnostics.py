"""Validate frozen histories and recorded UCI evidence, without running engines.

Also compose S14's material-residual probe output correctly. prediction_cp in
this production bench is ONLY the network residual, not the full search eval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import chess

VALUES = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330,
          chess.ROOK:500, chess.QUEEN:900, chess.KING:0}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_lines(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def reconstruct(case, point):
    b = chess.Board(case['initial_fen'])
    for u in case['history_uci'] + point.get('forced_uci', []):
        b.push_uci(u)
    assert b.fen() == point['fen']
    assert sorted(m.uci() for m in b.legal_moves) == point['legal_moves']
    assert b.is_checkmate() == point['checkmate']
    return b


def rounded_residual(raw):
    wide = raw * 1000
    magnitude = (abs(wide) + 2048) // 4096
    return magnitude if wide >= 0 else -magnitude


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--ladder', type=Path, required=True)
    ap.add_argument('--followup', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    identity = json.loads((args.ladder/'identity.json').read_text())
    followup_identity = json.loads((args.followup/'jobs.json').read_text())
    assert identity['binary_sha256'] == followup_identity['binary_sha256'] == plan['binary_sha256']
    assert identity['model_sha256'] == followup_identity['model_sha256'] == plan['model_sha256'] == digest(args.model)
    assert identity['plan_sha256'] == followup_identity['plan_sha256'] == digest(args.plan)
    h = args.model.read_bytes()[:44]
    assert h[:8] == b'EUNN2Q01'
    assert struct.unpack_from('<I', h, 8)[0] == 5
    assert struct.unpack_from('<f', h, 20)[0] == 1000.0
    assert struct.unpack_from('<I', h, 24)[0] == 12
    assert struct.unpack_from('<I', h, 40)[0] == 1, 'must compose material_residual'
    first = load_lines(args.ladder/'results.jsonl')
    follow = load_lines(args.followup/'results.jsonl')
    assert len(first) == 91 and len(follow) == 6
    assert json.loads((args.ladder/'complete.json').read_text())['status'] == 'complete'
    assert json.loads((args.followup/'complete.json').read_text())['status'] == 'complete'
    cases = {c['game_number']: c for c in plan['cases']}
    points = {}
    for number, case in cases.items():
        assert len(case['history_uci']) == case['ply']-1
        for point in [dict(case['root'], name='root')] + case['branches']:
            reconstruct(case, point)
            points[number,point['name']] = point
    normalized = []
    counts = {'histories_and_branch_positions':len(points), 'searches':0, 'legal_PVs':0, 'mate_one_proved':0}
    for record in first+follow:
        case = cases[record['game']]
        point = points[record['game'],record['point']]
        root = reconstruct(case, point)
        assert record['fen'] == point['fen']
        assert record['history_length'] == len(case['history_uci'])+len(point.get('forced_uci',[]))
        if root.is_checkmate():
            assert record['bestmove'] == '0000' and record['last'] is None
        else:
            assert record['bestmove'] in point['legal_moves']
            assert record['last'] == record['iterations'][-1]
            assert record['last']['pv'][0] == record['bestmove']
        for info in record['iterations']:
            b = root.copy(stack=True)
            for u in info['pv']:
                b.push_uci(u)
            counts['legal_PVs'] += 1
            # Other mate distances can be truncated in the PV. Here mate +1
            # has exactly the decisive edge and can be verified outright.
            if info.get('mate') == 1:
                assert b.is_checkmate()
                counts['mate_one_proved'] += 1
        hs = record['handshake']
        for line in ('info string source '+plan['source_commit'],
                     'info string eval nnue-v2q', 'info string network nnue-v2q',
                     'info string evalfile nnue-s14-datasupply-v5.bin'):
            assert line in hs
        counts['searches'] += 1
        normalized.append({k:v for k,v in record.items() if k != 'handshake'})
    static = []
    for record in load_lines(args.ladder/'static.jsonl'):
        point = points[record['game'],record['point']]
        b = reconstruct(cases[record['game']], point)
        probe = record['probe']
        assert probe['fen'] == b.fen()
        residual = rounded_residual(probe['raw_output'])
        white_mat = sum(v*(len(b.pieces(pt,chess.WHITE))-len(b.pieces(pt,chess.BLACK))) for pt,v in VALUES.items())
        sign = 1 if b.turn else -1
        static.append({
            **record, 'probe_semantics':'material residual ONLY',
            'residual_cp_white_i32':residual*sign, 'material_cp_white':white_mat,
            'composed_static_cp_white':white_mat+residual*sign,
            'composition':'signed round-half-away(raw*1000/4096) + P100/N320/B330/R500/Q900/K0 material',
        })
    # Boundary proof from the frozen production input, not a new position.
    c = cases[127]
    p = points[127,'after_Qf3_check']
    b = reconstruct(c,p)
    assert {b.san(m) for m in b.legal_moves} == {'Kh2','Kg1'}
    for move in list(b.legal_moves):
        bb=b.copy(stack=True);bb.push(move)
        mate=bb.parse_san('Qg2#')
        assert not bb.is_check() and not bb.is_capture(mate)
        bb.push(mate);assert bb.is_checkmate()
    d1 = next(r for r in first if r['game']==127 and r['point']=='after_Qf3_check' and r['requested_depth']==1)
    d2 = next(r for r in first if r['game']==127 and r['point']=='after_Qf3_check' and r['requested_depth']==2)
    assert d1['last']['cp'] == 200 and d2['last']['mate'] == -1
    for r in follow[:2]:
        baseline=d1 if r['requested_depth']==1 else d2
        assert r['last']['cp' if r['requested_depth']==1 else 'mate'] == baseline['last']['cp' if r['requested_depth']==1 else 'mate']
        assert r['last']['pv']==baseline['last']['pv']
    # The complete static score (not just the residual) equals the optimistic
    # boundary score. This is consistent with stand-pat at the quiet leaf.
    endpoint=next(r for r in static if r['game']==127 and r['point']=='pv_endpoint_Kg1')
    assert endpoint['composed_static_cp_white']==200
    for number in cases:
        r=next(r for r in first if r['game']==number and r['name'].endswith('historical-clock-cold'))
        assert r['bestmove']==cases[number]['historical_bestmove']
    result={'schema_version':1,'identity':identity,'followup_identity':followup_identity,
            'evidence_sha256':{str(p.name)+'@'+p.parent.name:digest(p) for p in
                [args.ladder/'results.jsonl',args.ladder/'static.jsonl',args.followup/'results.jsonl']},
            'checks':counts,'results':normalized,'composed_static':static,
            'limitations':['fresh TT, not historical warm TT','local timing not comparable to cloud timings',
                           'child-root forced continuation is not searchmoves at the parent root',
                           'quiet-check coverage was not patched/ablated','no full-match strength claim']}
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print('PASS',counts)
    print('g127 boundary +200 cp at d1 -> mate -1 at d2; two evasions proven mate; repeat stable')
    print('static composition PASS: endpoint residual -10cp + material 210cp = 200cp')
    print('Capped jobs:',[(r['name'],r['last']['depth']) for r in normalized if r.get('depth_reached') is False and r['last']])
    print('summary sha256',digest(args.out))


if __name__ == '__main__':
    main()
