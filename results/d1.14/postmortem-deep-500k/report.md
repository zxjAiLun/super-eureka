# D1.14 Stockfish postmortem

- Phase: `deep-review`
- Positions analyzed: `150`
- Node limit per search: `500000`
- Score model: `centipawn loss plus explicit mate outcomes`
- WDL: `not requested and not used`

## Profile comparison

- `Current`: moves `78`, mean/median CPL `270.42` / `246.5`, blunders `39`, mate swings `25`, best-move match `0.02564102564102564`, mate categories `allowed_mate=3, losing_mate_accelerated=22, winning_mate_delayed=3`
- `CurrentLmr`: moves `72`, mean/median CPL `267.82978723404256` / `245`, blunders `37`, mate swings `18`, best-move match `0.06944444444444445`, mate categories `allowed_mate=7, losing_mate_accelerated=14, winning_mate_delayed=4`

## Common-position move agreement

- Paired common positions: `42` (same move `41`, different move `1`)
- Different-move comparable CP groups: `1`; candidate lower CPL `0`, baseline lower CPL `1`, equal `0`

## Top losses

- Game 719 ply 115 `Current` `d4`: best `d2d3 cp:0`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 166 ply 47 `CurrentLmr` `Kxh3`: best `g4h4 cp:22`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 745 ply 30 `Current` `h6`: best `g7g6 cp:176`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 419 ply 152 `CurrentLmr` `b4`: best `a4b4 cp:-692`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 616 ply 31 `CurrentLmr` `Qc4`: best `f7e7 cp:-739`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 345 ply 78 `Current` `a3`: best `b6b4 mate:-5`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 465 ply 12 `Current` `Nxc6`: best `f7f5 cp:-467`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 652 ply 51 `Current` `Bc4`: best `g1h1 cp:-298`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 71 ply 146 `Current` `Rb6`: best `c1h1 cp:-261`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 429 ply 30 `Current` `Nf5`: best `e8f7 cp:-616`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 243 ply 81 `Current` `Rxa2`: best `a4a5 cp:-758`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 3 ply 38 `Current` `Bd7`: best `g7f8 cp:-674`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 810 ply 43 `CurrentLmr` `Qb4`: best `b6c6 cp:-864`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 76 ply 54 `CurrentLmr` `Rc8`: best `a8e8 cp:-703`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 781 ply 21 `Current` `Qxa2`: best `d8e6 cp:-512`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 186 ply 73 `Current` `Kh2`: best `g1f1 cp:-560`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 611 ply 85 `CurrentLmr` `Rd4+`: best `g1d1 mate:-7`, played `mate:-4`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 393 ply 58 `Current` `Kg8`: best `g7f7 cp:-668`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 814 ply 64 `Current` `Bg3`: best `f2c5 cp:-864`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 371 ply 60 `Current` `f6`: best `g7g6 cp:-680`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 720 ply 174 `Current` `Kg1`: best `f1e1 cp:-1158`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 438 ply 76 `Current` `Nc4`: best `g3g6 cp:-892`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 248 ply 131 `Current` `Rxg7`: best `g6f6 mate:-11`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 241 ply 95 `CurrentLmr` `Rg7`: best `g6g5 mate:-6`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 402 ply 118 `CurrentLmr` `Kh5`: best `g5h6 mate:-4`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 185 ply 109 `CurrentLmr` `h6`: best `f6f7 cp:-922`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 501 ply 111 `CurrentLmr` `Kb1`: best `b2a3 mate:-5`, played `mate:-4`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 542 ply 45 `Current` `Kf2`: best `g1g2 mate:-8`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 108 ply 118 `CurrentLmr` `h4`: best `b4b3 mate:-5`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 613 ply 56 `Current` `Re5`: best `g6d6 cp:-659`, played `mate:-4`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 333 ply 78 `Current` `f5`: best `g7g6 mate:-6`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 778 ply 134 `Current` `Kd1`: best `c1b2 mate:-7`, played `mate:-5`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 399 ply 251 `CurrentLmr` `Kb8`: best `b7b6 mate:-6`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 741 ply 19 `Current` `Qb4`: best `c8c6 cp:-779`, played `mate:-8`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 742 ply 19 `CurrentLmr` `Qb4`: best `c8c6 cp:-779`, played `mate:-8`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 275 ply 107 `CurrentLmr` `h5`: best `g1f1 cp:-953`, played `mate:-5`, CPL `None`, class `mate-swing`, mate `losing_mate_accelerated`
- Game 789 ply 44 `CurrentLmr` `Qxf5`: best `d3f5 mate:3`, played `cp:843`, CPL `None`, class `mate-swing`, mate `winning_mate_delayed`
- Game 472 ply 66 `Current` `Qf6+`: best `f7f8 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `winning_mate_delayed`
- Game 419 ply 103 `Current` `Qe2+`: best `e1d2 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `winning_mate_delayed`
- Game 661 ply 93 `Current` `Kd3`: best `b2f2 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `winning_mate_delayed`
- Game 109 ply 161 `CurrentLmr` `b6`: best `e6d6 mate:4`, played `mate:5`, CPL `None`, class `mate-swing`, mate `winning_mate_delayed`
- Game 489 ply 70 `CurrentLmr` `Rh5`: best `e5h8 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `winning_mate_delayed`
- Game 475 ply 28 `CurrentLmr` `Qg4+`: best `h4g5 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `winning_mate_delayed`
- Game 607 ply 19 `Current` `Nf6`: best `d8e7 cp:17`, played `cp:-624`, CPL `641`, class `blunder`, mate `None`
- Game 608 ply 19 `CurrentLmr` `Nf6`: best `d8e7 cp:17`, played `cp:-624`, CPL `641`, class `blunder`, mate `None`
- Game 143 ply 7 `Current` `d4`: best `a7a6 cp:221`, played `cp:-328`, CPL `549`, class `blunder`, mate `None`
- Game 144 ply 7 `CurrentLmr` `d4`: best `a7a6 cp:221`, played `cp:-328`, CPL `549`, class `blunder`, mate `None`
- Game 133 ply 59 `Current` `Qb6`: best `c7d7 cp:-307`, played `cp:-829`, CPL `522`, class `blunder`, mate `None`
- Game 293 ply 39 `CurrentLmr` `Qe2`: best `a1d1 cp:307`, played `cp:-212`, CPL `519`, class `blunder`, mate `None`
- Game 715 ply 24 `Current` `Bc7+`: best `c5e6 cp:-13`, played `cp:-486`, CPL `473`, class `blunder`, mate `None`
