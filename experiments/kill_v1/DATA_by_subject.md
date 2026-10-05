# MMLU-Redux 2.0: errors per subject

Pinned revision `372ea425445d51e1ba1188c56e5e893f8138621f`, counted from the Arrow files on 2026-10-06.
"key errors" = `wrong_groundtruth` + `no_correct_answer` + `multiple_correct_answers`. See `DATA.md`.

| subject | supercat | ok | wrong_gt | no_correct | multi_correct | **key errors** | bad_q | bad_opt | expert | any error |
|---|---|---|---|---|---|---|---|---|---|---|
| abstract_algebra | STEM | 89 | 6 | 2 | 0 | **8** | 2 | 0 | 1 | 11 |
| anatomy | other | 99 | 1 | 0 | 0 | **1** | 0 | 0 | 0 | 1 |
| astronomy | STEM | 94 | 1 | 1 | 0 | **2** | 2 | 2 | 0 | 6 |
| business_ethics | other | 85 | 0 | 0 | 0 | **0** | 14 | 0 | 1 | 15 |
| clinical_knowledge | other | 99 | 0 | 0 | 0 | **0** | 1 | 0 | 0 | 1 |
| college_biology | STEM | 98 | 1 | 0 | 0 | **1** | 1 | 0 | 0 | 2 |
| college_chemistry | STEM | 75 | 21 | 2 | 0 | **23** | 2 | 0 | 0 | 25 |
| college_computer_science | STEM | 97 | 0 | 2 | 1 | **3** | 0 | 0 | 0 | 3 |
| college_mathematics | STEM | 99 | 0 | 0 | 0 | **0** | 1 | 0 | 0 | 1 |
| college_medicine | other | 87 | 1 | 1 | 0 | **2** | 2 | 0 | 9 | 13 |
| college_physics | STEM | 100 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| computer_security | STEM | 97 | 3 | 0 | 0 | **3** | 0 | 0 | 0 | 3 |
| conceptual_physics | STEM | 92 | 4 | 0 | 1 | **5** | 0 | 0 | 3 | 8 |
| econometrics | social sci | 97 | 0 | 0 | 0 | **0** | 3 | 0 | 0 | 3 |
| electrical_engineering | STEM | 98 | 1 | 0 | 0 | **1** | 1 | 0 | 0 | 2 |
| elementary_mathematics | STEM | 97 | 2 | 0 | 0 | **2** | 1 | 0 | 0 | 3 |
| formal_logic | humanities | 87 | 1 | 2 | 9 | **12** | 1 | 0 | 0 | 13 |
| global_facts | other | 88 | 5 | 4 | 0 | **9** | 2 | 1 | 0 | 12 |
| high_school_biology | STEM | 95 | 2 | 1 | 0 | **3** | 1 | 1 | 0 | 5 |
| high_school_chemistry | STEM | 99 | 0 | 0 | 0 | **0** | 1 | 0 | 0 | 1 |
| high_school_computer_science | STEM | 100 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| high_school_european_history | humanities | 91 | 1 | 0 | 0 | **1** | 4 | 4 | 0 | 9 |
| high_school_geography | social sci | 100 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| high_school_government_and_politics | social sci | 100 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| high_school_macroeconomics | social sci | 97 | 0 | 0 | 1 | **1** | 2 | 0 | 0 | 3 |
| high_school_mathematics | STEM | 100 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| high_school_microeconomics | social sci | 98 | 1 | 0 | 0 | **1** | 1 | 0 | 0 | 2 |
| high_school_physics | STEM | 97 | 2 | 1 | 0 | **3** | 0 | 0 | 0 | 3 |
| high_school_psychology | social sci | 96 | 0 | 0 | 0 | **0** | 4 | 0 | 0 | 4 |
| high_school_statistics | STEM | 98 | 2 | 0 | 0 | **2** | 0 | 0 | 0 | 2 |
| high_school_us_history | humanities | 100 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| high_school_world_history | humanities | 99 | 1 | 0 | 0 | **1** | 0 | 0 | 0 | 1 |
| human_aging | other | 87 | 0 | 0 | 0 | **0** | 12 | 0 | 1 | 13 |
| human_sexuality | social sci | 81 | 2 | 2 | 1 | **5** | 11 | 1 | 2 | 19 |
| international_law | humanities | 95 | 2 | 0 | 3 | **5** | 0 | 0 | 0 | 5 |
| jurisprudence | humanities | 99 | 0 | 0 | 0 | **0** | 0 | 1 | 0 | 1 |
| logical_fallacies | humanities | 74 | 3 | 4 | 3 | **10** | 14 | 2 | 0 | 26 |
| machine_learning | STEM | 89 | 2 | 0 | 1 | **3** | 3 | 5 | 0 | 11 |
| management | other | 99 | 0 | 0 | 0 | **0** | 0 | 0 | 1 | 1 |
| marketing | other | 96 | 0 | 0 | 0 | **0** | 2 | 2 | 0 | 4 |
| medical_genetics | other | 100 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| miscellaneous | other | 90 | 0 | 4 | 1 | **5** | 4 | 1 | 0 | 10 |
| moral_disputes | humanities | 96 | 1 | 0 | 0 | **1** | 3 | 0 | 0 | 4 |
| moral_scenarios | humanities | 97 | 0 | 0 | 0 | **0** | 3 | 0 | 0 | 3 |
| nutrition | other | 97 | 0 | 0 | 0 | **0** | 1 | 1 | 1 | 3 |
| philosophy | humanities | 89 | 0 | 0 | 0 | **0** | 0 | 0 | 11 | 11 |
| prehistory | humanities | 100 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| professional_accounting | other | 92 | 0 | 5 | 0 | **5** | 0 | 1 | 2 | 8 |
| professional_law | humanities | 82 | 2 | 0 | 11 | **13** | 4 | 1 | 0 | 18 |
| professional_medicine | other | 99 | 0 | 0 | 0 | **0** | 1 | 0 | 0 | 1 |
| professional_psychology | social sci | 96 | 2 | 1 | 0 | **3** | 1 | 0 | 0 | 4 |
| public_relations | social sci | 91 | 3 | 0 | 3 | **6** | 3 | 0 | 0 | 9 |
| security_studies | social sci | 94 | 0 | 0 | 0 | **0** | 6 | 0 | 0 | 6 |
| sociology | social sci | 98 | 0 | 0 | 0 | **0** | 2 | 0 | 0 | 2 |
| us_foreign_policy | social sci | 99 | 0 | 0 | 0 | **0** | 1 | 0 | 0 | 1 |
| virology | other | 43 | 33 | 4 | 4 | **41** | 14 | 2 | 0 | 57 |
| world_religions | humanities | 99 | 0 | 0 | 0 | **0** | 1 | 0 | 0 | 1 |
| **all 57** | — | 5330 | 106 | 36 | 39 | **181** | 132 | 25 | 32 | 370 |
