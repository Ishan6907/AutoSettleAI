# Indian Car Accident Dataset: Evaluation & Labeling Report

- **Total Images Evaluated**: 254
- **Photos with Confirmed Vehicle Parts**: 184
- **Photos with Detected Structural Damages**: 137
- **Total Vehicle Parts Segmented**: 1096
- **Total Damage Instances Localized**: 278
- **VGGT 3D Grounding Verified Dents**: 117
- **Average Inference Latency**: 711.2 ms/image

### Damage Class Breakdown
| Damage Class | Instances Detected |
|---|---|
| Broken part | 159 |
| Dent | 119 |

### Most Damaged Vehicle Parts
| Vehicle Part | Damaged Instances |
|---|---|
| Windshield | 95 |
| Front-door | 47 |
| Front-bumper | 33 |
| Headlight | 28 |
| Back-door | 20 |
| Front-wheel | 20 |
| Back-bumper | 9 |
| Fender | 8 |
| Hood | 7 |
| Trunk | 4 |

### Sample Detected Accidents (Top Highlights)
| Image | Dimensions | Parts | Damages Detected | Annotated Output |
|---|---|---|---|---|
| `in_acc_001_file_crashed_taxi_multiple_car_accident_.jpg` | 2048x1536 | 14 | Back-door:Dent:minor(conf=0.57 [3D-Dent]);Back-bumper:Dent:moderate(conf=0.38 [3D-Dent]) | [`annotated/labeled_in_acc_001_file_crashed_taxi_multiple_car_accident_.jpg`](annotated/labeled_in_acc_001_file_crashed_taxi_multiple_car_accident_.jpg) |
| `in_acc_004_file_multiple_car_accident_rabindra_sada.jpg` | 2048x1536 | 14 | Hood:Dent:minor(conf=0.98 [3D-Dent]) | [`annotated/labeled_in_acc_004_file_multiple_car_accident_rabindra_sada.jpg`](annotated/labeled_in_acc_004_file_multiple_car_accident_rabindra_sada.jpg) |
| `in_acc_008_file_alto_800_accident_jpg.jpg` | 3264x1836 | 18 | Windshield:Broken part:severe(conf=0.95);Headlight:Broken part:severe(conf=0.88);Hood:Dent:moderate(conf=0.63 [3D-Dent]);Hood:Dent:moderate(conf=0.78 [3D-Dent]);Headlight:Broken part:moderate(conf=0.50);Windshield:Broken part:minor(conf=0.25);Windshield:Broken part:minor(conf=0.21);Headlight:Broken part:minor(conf=0.38);Headlight:Broken part:severe(conf=0.87);Front-bumper:Dent:minor(conf=0.48 [3D-Dent]);Hood:Dent:minor(conf=0.44 [3D-Dent]) | [`annotated/labeled_in_acc_008_file_alto_800_accident_jpg.jpg`](annotated/labeled_in_acc_008_file_alto_800_accident_jpg.jpg) |
| `in_acc_011_file_indian_navy_sub_lieutenant_atul_kum.jpg` | 700x464 | 3 | Headlight:Broken part:severe(conf=0.92) | [`annotated/labeled_in_acc_011_file_indian_navy_sub_lieutenant_atul_kum.jpg`](annotated/labeled_in_acc_011_file_indian_navy_sub_lieutenant_atul_kum.jpg) |
| `in_acc_012_file_indian_navy_sub_lieutenant_atul_kum.jpg` | 700x464 | 1 | Headlight:Broken part:severe(conf=0.68) | [`annotated/labeled_in_acc_012_file_indian_navy_sub_lieutenant_atul_kum.jpg`](annotated/labeled_in_acc_012_file_indian_navy_sub_lieutenant_atul_kum.jpg) |
| `in_acc_013_file_nitin_gadkari_witnessing_the_exchan.jpg` | 2200x1345 | 5 | Headlight:Broken part:moderate(conf=0.21) | [`annotated/labeled_in_acc_013_file_nitin_gadkari_witnessing_the_exchan.jpg`](annotated/labeled_in_acc_013_file_nitin_gadkari_witnessing_the_exchan.jpg) |
| `in_acc_014_file_oscar_fernandes_the_chief_minister_.jpg` | 2200x1294 | 5 | Headlight:Broken part:severe(conf=0.27) | [`annotated/labeled_in_acc_014_file_oscar_fernandes_the_chief_minister_.jpg`](annotated/labeled_in_acc_014_file_oscar_fernandes_the_chief_minister_.jpg) |
| `in_acc_015_file_oscar_fernandes_addressing_at_the_l.jpg` | 2200x1722 | 2 | Front-door:Dent:severe(conf=0.19 [2D-Flat]);Quarter-panel:Dent:minor(conf=0.46 [3D-Dent]) | [`annotated/labeled_in_acc_015_file_oscar_fernandes_addressing_at_the_l.jpg`](annotated/labeled_in_acc_015_file_oscar_fernandes_addressing_at_the_l.jpg) |
| `in_acc_017_file_oscar_fernandes_along_with_the_chie.jpg` | 2200x1203 | 10 | Headlight:Broken part:severe(conf=0.73);Back-bumper:Dent:minor(conf=0.80 [3D-Dent]);Back-bumper:Dent:minor(conf=0.69 [3D-Dent]);Headlight:Broken part:severe(conf=0.94) | [`annotated/labeled_in_acc_017_file_oscar_fernandes_along_with_the_chie.jpg`](annotated/labeled_in_acc_017_file_oscar_fernandes_along_with_the_chie.jpg) |
| `in_acc_019_file_oscar_fernandes_launching_the_pilot.jpg` | 2200x1461 | 14 | Back-door:Dent:minor(conf=0.55 [3D-Dent]) | [`annotated/labeled_in_acc_019_file_oscar_fernandes_launching_the_pilot.jpg`](annotated/labeled_in_acc_019_file_oscar_fernandes_launching_the_pilot.jpg) |
| `in_acc_020_file_sadhak_anshit_jpg.jpg` | 2737x2737 | 1 | Windshield:Broken part:minor(conf=0.58);Windshield:Broken part:minor(conf=0.37) | [`annotated/labeled_in_acc_020_file_sadhak_anshit_jpg.jpg`](annotated/labeled_in_acc_020_file_sadhak_anshit_jpg.jpg) |
| `in_acc_021_file_truck_collision_with_meridian_on_nh.jpg` | 4000x3000 | 3 | Front-wheel:Broken part:minor(conf=0.92);Back-wheel:Broken part:severe(conf=0.88);Back-wheel:Broken part:severe(conf=0.67) | [`annotated/labeled_in_acc_021_file_truck_collision_with_meridian_on_nh.jpg`](annotated/labeled_in_acc_021_file_truck_collision_with_meridian_on_nh.jpg) |
| `in_acc_027_file_direx_15_4_jpg.jpg` | 602x337 | 1 | Front-door:Dent:minor(conf=0.59 [3D-Dent]) | [`annotated/labeled_in_acc_027_file_direx_15_4_jpg.jpg`](annotated/labeled_in_acc_027_file_direx_15_4_jpg.jpg) |
| `in_acc_028_file_direx_15_5_jpg.jpg` | 602x337 | 3 | Headlight:Broken part:severe(conf=0.25) | [`annotated/labeled_in_acc_028_file_direx_15_5_jpg.jpg`](annotated/labeled_in_acc_028_file_direx_15_5_jpg.jpg) |
| `in_acc_029_file_ships_in_line_formation_during_dire.jpg` | 600x335 | 0 | Front-door:Dent:severe(conf=0.55 [3D-Dent]) | [`annotated/labeled_in_acc_029_file_ships_in_line_formation_during_dire.jpg`](annotated/labeled_in_acc_029_file_ships_in_line_formation_during_dire.jpg) |
