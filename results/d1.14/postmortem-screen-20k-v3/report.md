# D1.14 Stockfish postmortem

- Phase: `screen`
- Positions analyzed: `1000`
- Node limit per search: `20000`
- Score model: `centipawn loss plus explicit mate outcomes`
- WDL: `not requested and not used`

## Profile comparison

- `Current`: moves `506`, mean/median CPL `60.471014492753625` / `22.0`, blunders `43`, mate swings `26`, best-move match `0.38735177865612647`
- `CurrentLmr`: moves `494`, mean/median CPL `59.71638141809291` / `22`, blunders `42`, mate swings `21`, best-move match `0.3967611336032389`

## Top losses

- Game 719 ply 115 `Current` `d4`: best `c8e8 cp:17`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 166 ply 47 `CurrentLmr` `Kxh3`: best `g4h4 cp:90`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 745 ply 30 `Current` `h6`: best `g7g6 cp:162`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 419 ply 152 `CurrentLmr` `b4`: best `a4b4 cp:-592`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 616 ply 31 `CurrentLmr` `Qc4`: best `f7f6 cp:-605`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 345 ply 78 `Current` `a3`: best `b6b4 cp:-801`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 465 ply 12 `Current` `Nxc6`: best `f7f5 cp:-494`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 652 ply 51 `Current` `Bc4`: best `g1h1 cp:-206`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 71 ply 146 `Current` `Rb6`: best `c1h1 cp:-226`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 429 ply 30 `Current` `Nf5`: best `e8f7 cp:-510`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 243 ply 81 `Current` `Rxa2`: best `a4a3 cp:-486`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 3 ply 38 `Current` `Bd7`: best `c8g4 cp:-527`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 810 ply 43 `CurrentLmr` `Qb4`: best `b6c6 cp:-658`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 76 ply 54 `CurrentLmr` `Rc8`: best `a8e8 cp:-634`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 781 ply 21 `Current` `Qxa2`: best `d8e6 cp:-521`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 186 ply 73 `Current` `Kh2`: best `g1f1 cp:-516`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 611 ply 85 `CurrentLmr` `Rd4+`: best `g1d1 cp:-856`, played `mate:-4`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 409 ply 121 `Current` `Kc8`: best `d8c8 cp:-917`, played `mate:-5`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 393 ply 58 `Current` `Kg8`: best `g7f7 cp:-435`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 814 ply 64 `Current` `Bg3`: best `f2c5 cp:-649`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 371 ply 60 `Current` `f6`: best `g7g6 cp:-427`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 720 ply 174 `Current` `Kg1`: best `h6h7 cp:-610`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 438 ply 76 `Current` `Nc4`: best `g3g6 cp:-729`, played `mate:-2`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 248 ply 131 `Current` `Rxg7`: best `g6f6 cp:-812`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 241 ply 95 `CurrentLmr` `Rg7`: best `g6g5 cp:-1041`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 402 ply 118 `CurrentLmr` `Kh5`: best `g5h4 cp:-681`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 311 ply 210 `CurrentLmr` `Ka8`: best `b8a7 cp:-556`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 185 ply 109 `CurrentLmr` `h6`: best `f6f7 cp:-683`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 501 ply 111 `CurrentLmr` `Kb1`: best `b2c1 cp:-974`, played `mate:-4`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 542 ply 45 `Current` `Kf2`: best `a1d1 cp:-746`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 108 ply 118 `CurrentLmr` `h4`: best `b4b3 cp:-769`, played `mate:-9`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 613 ply 56 `Current` `Re5`: best `g6d6 cp:-590`, played `mate:-4`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 333 ply 78 `Current` `f5`: best `g7g6 cp:-748`, played `mate:-3`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 778 ply 134 `Current` `Kd1`: best `c1b2 cp:-879`, played `mate:-5`, CPL `None`, class `mate-swing`, mate `catastrophic_mate_swing`
- Game 789 ply 44 `CurrentLmr` `Qxf5`: best `d3f5 mate:3`, played `cp:711`, CPL `None`, class `mate-swing`, mate `missed_mate`
- Game 491 ply 70 `Current` `Bxh3`: best `c8h3 mate:3`, played `cp:711`, CPL `None`, class `mate-swing`, mate `missed_mate`
- Game 227 ply 156 `CurrentLmr` `Kc4`: best `h3h4 mate:6`, played `cp:722`, CPL `None`, class `mate-swing`, mate `missed_mate`
- Game 675 ply 135 `CurrentLmr` `Qh4+`: best `f2h4 mate:3`, played `cp:559`, CPL `None`, class `mate-swing`, mate `missed_mate`
- Game 266 ply 188 `CurrentLmr` `Rd2+`: best `d3d2 mate:4`, played `cp:633`, CPL `None`, class `mate-swing`, mate `missed_mate`
- Game 676 ply 154 `CurrentLmr` `Kf2`: best `e3f2 mate:4`, played `cp:1274`, CPL `None`, class `mate-swing`, mate `missed_mate`
- Game 472 ply 66 `Current` `Qf6+`: best `f7f8 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `mate_distance_increase`
- Game 419 ply 103 `Current` `Qe2+`: best `e1d2 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `mate_distance_increase`
- Game 661 ply 93 `Current` `Kd3`: best `b2f2 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `mate_distance_increase`
- Game 109 ply 161 `CurrentLmr` `b6`: best `e6d6 mate:4`, played `mate:5`, CPL `None`, class `mate-swing`, mate `mate_distance_increase`
- Game 489 ply 70 `CurrentLmr` `Rh5`: best `e5h8 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `mate_distance_increase`
- Game 775 ply 115 `CurrentLmr` `Kxh5`: best `g5g6 mate:4`, played `mate:13`, CPL `None`, class `mate-swing`, mate `mate_distance_increase`
- Game 475 ply 28 `CurrentLmr` `Qg4+`: best `h4g5 mate:3`, played `mate:4`, CPL `None`, class `mate-swing`, mate `mate_distance_increase`
- Game 607 ply 19 `Current` `Nf6`: best `d8e7 cp:-62`, played `cp:-625`, CPL `563`, class `blunder`, mate `None`
- Game 608 ply 19 `CurrentLmr` `Nf6`: best `d8e7 cp:-62`, played `cp:-625`, CPL `563`, class `blunder`, mate `None`
- Game 151 ply 137 `CurrentLmr` `Qf6+`: best `c2d2 cp:1134`, played `cp:618`, CPL `516`, class `blunder`, mate `None`
