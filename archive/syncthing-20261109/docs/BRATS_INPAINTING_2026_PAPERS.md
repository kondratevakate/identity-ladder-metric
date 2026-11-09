# Полный каталог BraTS-Inpainting MICCAI 2026

Проверено 2 ноября 2026. Все 16 camera-ready PDF из [раздела proceedings](https://papers.miccai.org/miccai-2026-sat/categories/#brats-inpainting) получены; изучены методы, results и ограничения. Состав источников и хеши: [manifest](BRATS_INPAINTING_2026_SOURCES.json). Основной обзор: [состояние задачи](BRATS_INPAINTING_STATE_OF_THE_ART_2026-11-02.md).

Ниже mean metrics, как они приведены авторами; PSNR в dB, MSE без пересчёта из RMSE. Таблица также доступна в [CSV](BRATS_INPAINTING_2026_RESULTS.csv). «Official validation» означает, что авторы сообщают ответ сервера; независимо получить server row не удалось. Разный preprocessing, набор случаев и версия scorer могут менять результаты. Таблица следует порядку proceedings и не задаёт итоговый rank.

| ID, работа и источник PDF | Evaluation | SSIM | PSNR | MSE | Метод и граница вывода |
|---|---|---:|---:|---:|---|
| [002 Enhanced 3D U-Net](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_002.pdf) | Local validation, 126 | 0.9065 | 27.62 | 0.00297 | Attention, deep supervision, residual blocks, Sobel loss; числа winners заимствованы из других публикаций, общий split не показан |
| [003 Zero-Shot 2.5D Flow Priors](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_003.pdf) | Official validation, 219; N=10 | 0.816 | 22.923 | 0.007 | Unconditional triplet prior + Restora-Flow; zero-shot относится к inpainting conditioning, prior обучается на MRI |
| [004 Conditional Flow Matching](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_004.pdf) | Official validation, 219 | 0.862 | 24.87 | 0.0047 | Conditioning, few-step sampler, tiling, flips, ensemble; последовательные ablations на challenge server являются development |
| [005 Dynamic Shape-Aware Pix2Pix](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_005.pdf) | Official hidden-validation, 219 | 0.780 | 21.581 | 0.009 | Two-stage conditional GAN + dynamic masks; local scores 0.804/20.552/0.013 имеют другой протокол |
| [007 Anatomy-Aware Healthy Brain Inpainting](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_007.pdf) | Official validation, direct model | 0.810923 | 22.6507 | 0.007373 | Input guidance, auxiliary tissue task и segmentation loss проверяются раздельно; SynthSeg-derived labels не являются ручным anatomical ground truth |
| [008 U-DiT / Healthy Attention](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_008.pdf) | Official validation, 219 | 0.864 | 24.71 | 0.0046 | Healthy-only attention и контралатеральный input; local final-model comparison на 60 случаях отделён от server evaluation |
| [009 SSIM-Prioritized Attention U-Net](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_009.pdf) | Table 3: validation; abstract: online test | 0.813832 | 23.4117 | 0.006416 | MS-SSIM, Haar loss, boundary penalties, EMA; неоднозначное имя evaluation stage сохранено |
| [010 Mask-Guided U-Net / waterdog](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_010.pdf) | Organizer validation, 219 | 0.798 | 22.340 | 0.008 | Component-wise overlapping tiles, 4-view TTA; авторы раскрывают отличие local intensity normalization и spatial support |
| [011 GAC-WaveDiff](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_011.pdf) | Online validation, Table 1 | 0.827 | 23.38 | 0.0062 | fastWDM3D + coordinates + distance transform + augmentation; 10.19 s/volume на RTX 5090, baseline 0.26 s; выигрыш не бесплатен по времени |
| [013 RARF](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_013.pdf) | Official validation; K=50 | 0.832 | 24.008 | 0.006 | Region-aware flow; submission checkpoint отличается от checkpoint исследования sampling strategies |
| [014 P3D-RAD](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_014.pdf) | Official validation, 219; Model B, 4-run mean | 0.832 | 24.45 | 0.006 | (2+1)D DDPM и region-aware schedule; опубликован downstream SynthSeg Dice, Table 1 |
| [016 Sharpening the Ensemble](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_016.pdf) | Official validation, 219; lambda=0.5 | 0.8572 | 25.10 | Не приведён для server row | Refiner ensemble; paired held-out gain мал, 10-seed spread превышает gain выбранного seed; base-model training overlap раскрыт |
| [017 Mask-Gated U-Net with Void-Size Conditioning](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_017.pdf) | Official validation, 219 | 0.832 | 23.366 | 0.006 | Gated skips, void-volume FiLM, background suppression; ablations локальные и последовательные; fine-detail blur сохраняется |
| [018 CATCH](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_018.pdf) | Official validation, 219; mean-N=5 | 0.772 | 20.89 | 0.0098 | Conditional Haar diffusion, weighted mask mixture; только selected pipeline оценён организаторами, controls имеют internal scores |
| [019 Retrieval-Augmented WDM](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_019.pdf) | Local val-150; N=8 + TTA | 0.8786 | 23.99 | 0.00408 | Retrieved donors + HCP prior; сравнение с no-prior diffusion на том же split, не official test; donor lesions могут попадать в prior |
| [020 RESIN](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_020.pdf) | Official validation, 219; 3-model ensemble | 0.8436 | 24.657 | 0.00541 | Full-resolution norm-free residual network; patient-grouped folds; architecture ablation меняет несколько свойств одновременно |

## Что добавляет каждая группа работ

**Регрессионные модели (002, 007–010, 017, 020).** Основные улучшения связаны с mask diversity, spatial context, full-resolution features, training schedule и inference averaging. Их результаты не поддерживают объяснение качества только размером backbone. 002 не демонстрирует общего test-set сравнения с winners. 010 особенно полезна как пример того, насколько меняются absolute scores при смене normalization/support.

**Генеративные модели (003, 004, 011, 013, 014, 018, 019).** Flow и diffusion используются с conditional context, region-aware noise, wavelet compression или retrieval prior. Несколько работ прямо выбирают average нескольких samples ради distortion metrics. 003 показывает N=1→10: SSIM 0.793→0.816, PSNR 21.888→22.923, MSE 0.009→0.007. В 011 два independent unflipped samples почти не помогают; основной inference gain связывается с flip-based TTA. Значит, любые улучшения ансамбля нельзя автоматически приписывать только stochastic averaging.

**Anatomical и downstream evaluation (007, 014).** 007 различает similarity и frozen-segmenter consistency: добавление segmentation objective повышает held-out SegDice с 64.22% до 65.80%; этот gain относится к автоматически полученному label space. 014 сообщает mean SynthSeg Dice 0.845, 0.853, 0.857 и 0.860 для averaging 1, 2, 3 и 4 runs. Это уже опубликованная проверка поведения downstream overlap при усреднении; она не измеряет истинность missing anatomy внутри опухоли.

**Refinement и retrieved prior (016, 019).** Refiner улучшает server SSIM с 0.8555 до 0.8572; прирост мал относительно вариации обучения. Retrieval prior в 019 даёт на val-150 SSIM 0.8786 против 0.8755 у diffusion без prior, PSNR 23.99 против 23.77 и MSE 0.00408 против 0.00425. Blurred-GT prior в той же работе является oracle upper bound и не допустим как deployable comparator.

## Какие ограничения повторяются

- Доступный healthy sub-mask позволяет проверять reconstruction; область actual lesion не имеет healthy reference. Глобальные заявления о восстановлении индивидуальной доопухолевой анатомии превышают данный benchmark.
- Большие voids труднее маленьких. Нужны size strata рядом с aggregate scores; размер и состав masks могут объяснять cross-split differences.
- Local validation, online validation и hidden final test нельзя объединять. 009 содержит конфликт label стадии уже внутри одной публикации.
- Mean±SD между cases не измеряет variability retraining. В 016 camera-ready прямо анализирует 10 seeds; многие остальные reported configurations не имеют такого анализа.
- Наблюдения о smoothness и выгоде averaging уже содержатся в 003, 004, 008, 013, 014, 016 и 017; их не следует объявлять отсутствующими в литературе.
- Из этих 16 публикаций нельзя вывести сравнительный рейтинг LPIPS/DISTS/HFEN или доказать superiority новой identity metric: это отдельный отсутствующий результат.

## Подтверждение чисел и версии

Основные точки извлечения: 002 Table 1; 003 Table 2; 004 Tables 1–3; 005 Table 3; 007 Tables 5–6; 008 Table 1; 009 Table 3; 010 Table 2; 011 Table 1; 013 Table 2; 014 Tables 1 и 3; 016 Table 2 и раздел 5; 017 Table 2; 018 abstract и Results; 019 Table 1; 020 Table 3. Для camera-ready 016 нумерация таблиц отличается от arXiv v1.

В 009 дополнительные error rows и их normalization/support требуют уточнения; в итоговую таблицу извлечены только SSIM, PSNR, MSE, а stage ambiguity сохранена. Для 002 заимствованные winner scores не перенесены в общую таблицу как парное сравнение.

Итоговые места команд см. отдельно в основном обзоре. Открытая полная численная таблица final hidden test 2026 не найдена. Поиск охватывал Challenge Portal/Synapse, официальный GitHub, Orchestrator и публикации top teams. Доступ к закрытым данным и новые model runs для этой фиксации не использовались.
