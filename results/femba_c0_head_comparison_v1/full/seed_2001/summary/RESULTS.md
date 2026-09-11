# FEMBA C0 无锚点下游比较

80 个新实验与 40 个已核验线性基线；seed=2001。

## vrq / state

|组别|下游|ACC %|BACC %|AUROC %|BCE|
|---|---|---:|---:|---:|---:|
|A3|fractional_dog_polykan|78.210|77.965|81.729|0.525|
|A4|fractional_dog_polykan|76.419|76.281|83.082|0.558|
|A3|linear|78.210|77.375|81.001|0.546|
|A4|linear|79.170|78.902|83.970|0.511|
|A3|mlp|78.908|78.529|81.807|0.550|
|A4|mlp|75.590|75.767|83.408|0.553|

## vrq / severity

|组别|下游|ACC %|BACC %|AUROC %|BCE|
|---|---|---:|---:|---:|---:|
|A3|fractional_dog_polykan|60.870|61.364|68.182|0.909|
|A4|fractional_dog_polykan|69.565|69.318|77.652|0.803|
|A3|linear|56.522|56.061|62.879|0.650|
|A4|linear|65.217|64.773|79.545|0.557|
|A3|mlp|60.870|60.606|63.636|1.262|
|A4|mlp|69.565|69.318|76.136|0.993|

## city / state

|组别|下游|ACC %|BACC %|AUROC %|BCE|
|---|---|---:|---:|---:|---:|
|A3|fractional_dog_polykan|77.149|78.976|84.751|0.477|
|A4|fractional_dog_polykan|76.721|77.295|84.037|0.495|
|A3|linear|76.624|77.812|84.093|0.475|
|A4|linear|78.829|80.274|85.506|0.510|
|A3|mlp|75.500|77.585|84.210|0.498|
|A4|mlp|76.367|77.299|83.557|0.529|

## city / severity

|组别|下游|ACC %|BACC %|AUROC %|BCE|
|---|---|---:|---:|---:|---:|
|A3|fractional_dog_polykan|68.831|68.831|71.513|0.757|
|A4|fractional_dog_polykan|69.481|69.481|73.646|0.842|
|A3|linear|74.675|74.675|79.271|0.558|
|A4|linear|68.182|68.182|73.107|0.709|
|A3|mlp|71.429|71.429|74.405|0.649|
|A4|mlp|63.636|63.636|71.816|0.794|

## dataset_macro / state

|组别|下游|ACC %|BACC %|AUROC %|BCE|
|---|---|---:|---:|---:|---:|
|A3|linear|77.417|77.594|82.547|0.511|
|A4|linear|79.000|79.588|84.738|0.510|
|A3|fractional_dog_polykan|77.679|78.470|83.240|0.501|
|A4|fractional_dog_polykan|76.570|76.788|83.559|0.526|
|A3|mlp|77.204|78.057|83.008|0.524|
|A4|mlp|75.978|76.533|83.483|0.541|

## dataset_macro / severity

|组别|下游|ACC %|BACC %|AUROC %|BCE|
|---|---|---:|---:|---:|---:|
|A3|linear|65.599|65.368|71.075|0.604|
|A4|linear|66.700|66.477|76.326|0.633|
|A3|fractional_dog_polykan|64.850|65.097|69.847|0.833|
|A4|fractional_dog_polykan|69.523|69.399|75.649|0.823|
|A3|mlp|66.149|66.017|69.021|0.956|
|A4|mlp|66.601|66.477|73.976|0.894|

## 解释边界

- Single seed and fixed training settings; numeric gains are not statistical significance.
- No anchors or known-rest calibration enter inference. Legacy C0 target exclusions remain.
- EA uses all unlabeled windows of each subject offline and is transductive, not strictly inductive.
- The complete pretraining sample manifest is missing; pretraining overlap has not been excluded.
- Design was informed by existing test results and is exploratory.
- Only pretrained frozen and finetuned encoders are tested; this does not establish a pretraining-versus-random effect.
- DoG-versus-MLP compares the complete mappings; it does not isolate fractional and DoG components.
- Same selection rules and maximum epochs, not equal actual updates or FLOPs.
