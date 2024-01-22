# Coevolution of norm psychology and cooperation through exapted conformity

Source Code of Y. Kido and M. Takezawa "Coevolution of norm psychology and cooperation through exapted conformity" in preparation.

## Models

|                                    |       Model 1        |         Model 2         |         Model 3         |
| ---------------------------------- | :------------------: | :---------------------: | :---------------------: |
| Assumed norm psychology            |      $\alpha_i$      | $\alpha_i$ & $\alpha_d$ | $\alpha_i$ & $\alpha_d$ |
| Initial distribution of $\alpha_i$ | $\sim U(0.00, 0.05)$ |  $\sim U(0.00, 0.05)$   |  $\sim U(0.00, 0.05)$   |
| Initial distribution of $\alpha_d$ |                      |  $\sim U(0.00, 0.05)$   | $\sim N(0.30, 0.25^2)$  |

## Requirements

requirements で必要なパッケージをインストールして。
pip install -r requirements.txt

## Model 1

We build a simple model of family groups, in which exchange of brides and resultant cooperation and competition are considered, by applying an agent-based model and multi-level evolution. By introducing one dimensional trait and preference, we show the emergence of the incest taboo.

Run [one-trait/emergy.py](one-trait/emerge.py) to get the phase diagrams on the emergence of the incest taboo.
引数の説明　 command line arguments 　　 run 番号
出力されるファイルの説明

## Model 2

By introducing two dimensional trait and preference, we show the emergence of kinship structures.

Run [two-trait/emergy.py](two-trait/emerge.py) to get the phase diagrams on the emergence of kinship structures.
引数の説明　 1 つ目に，run 番号
出力されるファイルの説明

## Model 3

We simplify the two-traiit model and discuss the evolution of kinship structures and descent systems.
Run [revised/structure.py](revised/structure.py) for the simulation.
引数の説明 1 つ目に，descriptive norm psychology initial distribution $\mu$ 　 0.1, 0.3, 0.5 の結果を SI で比較している
2 つ目に，run 番号
出力されるファイルの説明

## Supplementary Models

group size GROUP_SIZE
migration rate MIGRATION_RATE

## License

[MIT](LICENSE)
