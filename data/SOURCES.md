# Corpus sources

`news.txt` contains short text extracts, in this order. Each is attributed to the Wikinews contributors named in the article history. The links identify the source and its contributor history. Formatting was removed and the second extract was shortened. No photos, third-party quotations, or linked publications were copied.

| Title | Source | License |
| --- | --- | --- |
| Euro 2008: Austria vs. Poland | [Wikinews, June 12, 2008](https://en.wikinews.org/wiki/Euro_2008:_Austria_vs._Poland) | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) |
| New Zealand defeats South Africa to win 2024 women's T20 cricket world cup | [Wikinews, October 24, 2024](https://en.wikinews.org/wiki/New_Zealand_defeats_South_Africa_to_win_2024_women%27s_T20_cricket_world_cup) | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) |
| England defeats Australia and wins 2010 Twenty20 Cricket World Cup | [Wikinews, May 17, 2010](https://en.wikinews.org/wiki/England_defeats_Australia_and_wins_2010_Twenty20_Cricket_World_Cup) | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) |

[Wikinews license chronology](https://en.wikinews.org/wiki/Wikinews:2024_Copyright_license_upgrade). Only a few article pages were consulted manually through search. Training reads the checked-in files and makes no web requests.

`commentary.txt` is original synthetic play-by-play produced by `build_commentary.py`. These are fictional match situations and outcomes, released as [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Synthetic lines make up the bulk of the corpus. The validation split is drawn from those lines, with the three news extracts included in training.
