# BraTS inpainting: метрики, результаты и последние работы

Дата проверки: 2 ноября 2026. Область: Local Synthesis of Healthy Brain Tissue via Inpainting, T1n MRI. Это обзор опубликованного состояния задачи; новых экспериментов в нём нет.

## 1. Задача и границы доказательств

Модель получает voided T1n и маску, заполняет отсутствующую область здоровой тканью. Количественная оценка проводится на здоровой части скрытой области, где имеется reference. Она не подтверждает восстановление истинной доопухолевой анатомии внутри опухоли: такого reference обычно нет. Официальные материалы: [challenge 2026](https://github.com/BraTS-inpainting/2026_challenge/blob/main/evaluation/README.md), [обзор организаторов, v3](https://arxiv.org/html/2305.08992v3).

Проверены официальная документация, реестр BraTS Orchestrator, первичные статьи 2025–2026 и все 16 camera-ready PDF из каталога MICCAI 2026 BraTS-Inpainting. Для каждой работы извлечены постановка, опубликованный результат, evaluation stage и ограничения сравнения. Полный каталог и таблица чисел находятся в [приложении](BRATS_INPAINTING_2026_PAPERS.md); [manifest](BRATS_INPAINTING_2026_SOURCES.json) фиксирует URL и SHA-256 PDF. Это полный охват данного proceedings-раздела, но не систематический review всех MRI inpainting публикаций.

## 2. Какие метрики используются сейчас

| Уровень | Метрики | Что установлено |
|---|---|---|
| Официальная оценка 2026 | SSIM ↑, PSNR ↑, MSE ↓ | В документации 2026 перечислены эти три метрики; считаются внутри оцениваемой области |
| Исторический отчёт 2023 | SSIM ↑, PSNR ↑, RMSE ↓ | Результаты представлены медианами и IQR; нельзя напрямую сравнивать со средними новых статей |
| Дополнительные выходы пакета | MAE, MSLE, RMSE, варианты PSNR | Выходы инструмента не равны независимым критериям официального рейтинга |
| Исследовательская оценка MRI similarity | LPIPS, DISTS, HFEN, SSFD/RINFD и др. | Релевантные альтернативы из MRI IQA; их превосходство именно на BraTS inpainting пока не установлено приведёнными источниками |

Источники: [evaluation README 2026](https://github.com/BraTS-inpainting/2026_challenge/blob/main/evaluation/README.md), [официальный пакет](https://github.com/BraTS-inpainting/inpainting), [обзор challenge](https://arxiv.org/html/2305.08992v3).

Официальный rank-sum ранжирует методы по каждой метрике для каждого case и объединяет ранги с равными весами. При фиксированном диапазоне PSNR монотонно связан с MSE, а RMSE является корнем MSE. Поэтому набор сильно перекрывается по информации. Важны точная версия scorer, нормализация, clipping и реализация SSIM. Проверки версии 0.0.12 в этом проекте лежат в `src/brats_panel_checks.py`; это аудит реализации, а не независимое подтверждение рейтинга 2026.

«SOTA-метрики» здесь следует понимать как два разных факта: официальный набор, определяющий победителя challenge, и современные альтернативы для исследования качества. В проверенных источниках нет единой новой метрики, принятой как валидированный стандарт perceptual/anatomical quality именно BraTS inpainting.

## 3. Текущие места команд

По [официальной документации BraTS Orchestrator](https://brats.readthedocs.io/en/stable/reference/algorithms/#inpainting):

| Год | Место | Авторы | Идентификатор |
|---|---|---|---|
| 2026 | 1 | Kömürcü et al. | BraTS26_1 |
| 2026 | 2 | Hung et al. | BraTS26_2A |
| 2026 | 2 | Cap et al. | BraTS26_2B |
| 2026 | 3 | Persson et al. | BraTS26_3 |
| 2026 | 4 | Danese et al. | BraTS26_4A |
| 2026 | 4 | Karimaghaloo et al. | BraTS26_4B |
| 2025 | 1 | Zhang et al.; Ferreira et al. | BraTS25_1A; BraTS25_1B |
| 2024 | 1 | Zhang et al. | BraTS24_1 |
| 2023 | 1 | Zhang / Ying-Weng et al. | BraTS23_1 |

Это места из реестра алгоритмов. Они не восстановлены из средних чисел следующей таблицы. В реестре для новых entries ссылка на paper отмечена N/A; совпадение авторов с публикацией не удостоверяет тождество опубликованной конфигурации финальному контейнеру. Полная таблица численных результатов hidden test 2026 здесь не подтверждена.

## 4. Опубликованные численные результаты

Результаты возвращены сервером по сообщению соответствующих авторов; прямое независимое чтение строк Synapse leaderboard не выполнено. Таблица не сортируется как единый рейтинг.

| Метод / источник | Evaluation set | SSIM | PSNR, dB | Error | Ограничение |
|---|---|---:|---:|---:|---|
| Zhang et al., 2025 | Final online test | 0.91928 ± 0.08844 | 26.932 ± 5.057 | RMSE 0.05163 ± 0.02611 | Mean ± SD, Table 2 [A] |
| U-DiT, Danese et al., 2026 | Official validation, 219 | 0.864 ± 0.085 | 24.71 ± 4.03 | MSE 0.0046 ± 0.0032 | Mean ± SD; arXiv Table 1 [B] |
| Conditional Flow Matching, Hung et al., 2026 | Official validation, 219 | 0.862 ± 0.090 | 24.87 ± 4.55 | MSE 0.0047 ± 0.0035 | Camera-ready Table 3 [C] |
| Ensemble + SSIM refiner, Kömürcü & Öksüz, 2026 | Official validation, 219 | 0.8572 | 25.10 | Не приведён в этой таблице статьи | Lambda 0.5, arXiv Table 3 [D] |
| RESIN, 3-model ensemble, 2026 | Official validation, 219 | 0.8436 | 24.657 | MSE 0.00541 | Camera-ready Table 3 [E] |
| RARF, K=50, 2026 | Official validation | 0.832 | 24.008 | MSE 0.006 | Отдельный submission checkpoint, Table 2 [F] |
| fastWDM3D, 2025 | BraTS test set в статье | 0.8571 | 22.26 | MSE 0.0079 | Не отождествлять с final test 2025; 2 steps, 1.81 s/image [G] |
| Retrieval-augmented WDM, Persson et al., 2026 | Internal held-out | 0.879 | Не извлечён | Не извлечён | Число из abstract; не official validation [H] |

[A] [Zhang, Weng, Chen, Table 2](https://arxiv.org/html/2511.20202v1). [B] [Danese et al., Table 1](https://arxiv.org/html/2607.27974v1). [C] [Hung et al., camera-ready PDF](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_004.pdf). [D] [Kömürcü & Öksüz, Table 3](https://arxiv.org/html/2609.03981v1). [E] [Cap, camera-ready PDF](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_Inpainting_020.pdf). [F] [RARF, Table 2](https://arxiv.org/html/2609.03956v1). [G] [fastWDM3D](https://arxiv.org/html/2507.13146v1). [H] [Persson et al.](https://papers.miccai.org/miccai-2026-sat/BraTS_Inpainting_019.html).

Для сопоставления старых winners на одном validation set Cap, Table 3, приводит: winner 2023 SSIM 0.812 / PSNR 21.446 / MSE 0.009; winner 2024 0.841 / 23.381 / 0.007; winner 2025 0.873 / 24.996 / 0.00476. Это пересказ опубликованных prior-winner результатов в статье Cap, а не новая независимая проверка. Он показывает, почему низшее численное SSIM одной публикации не опровергает место команды на другом этапе challenge. Нельзя получать mean MSE возведением mean RMSE в квадрат.

## 5. Обзор последних работ по inpainting

### Zhang, Weng & Chen, 2025: Robust 3D Brain MRI Inpainting with Random Masking Augmentation

3D U-Net, MAE + SSIM, пять вариантов healthy masks на scan. Статья сообщает первое место 2025 и final-test числа [A]. В актуальном Orchestrator первое место также указано для Ferreira. Главный вклад работы: качество обычной регрессии существенно зависит от разнообразия тренировочных масок. В статье отмечена остаточная размытость. Её validation и test таблицы используют разные error labels; численные значения нужно сохранять с исходными обозначениями.

### Durrer et al., 2025: fastWDM3D

Ускоренная wavelet diffusion с variance-preserving schedule и reconstruction losses. Два шага и 1.81 s/image при результатах [G]. Важен компромисс скорости и качества; опубликован [код](https://github.com/AliciaDurrer/fastWDM3D). Сравнение времени зависит от hardware и sampling configuration. Работа показывает развитие быстрых generative approaches; не устанавливает их победу в рейтинге 2026.

### Danese, Lombardi & Di Noia, 2026: Now You Have My Healthy Attention

U-DiT использует healthy-only attention и противоположное полушарие как patient-specific context; mirrored input сопровождается validity map. Авторы рассматривают deterministic regression под distortion-based scoring. Результат [B] относится к server validation; локальные архитектурные ablations имеют другой split. Работа прямо обсуждает smoothness и anatomical realism. Доступен [авторский код](https://github.com/sisinflab/WaveUDiT).

### Hung et al., 2026: Conditional Flow Matching

Camera-ready [C] описывает conditioning окружающей анатомией, обучение на healthy-region mask, tile blending, flip TTA и averaging. На official validation single-sample baseline даёт SSIM 0.741 / PSNR 19.87, финальный recipe 0.862 / 24.87. Это совокупный эффект нескольких изменений, не чистая оценка усреднения. Важное ограничение: ablations являются последовательными challenge submissions, поэтому validation используется для выбора. Сам тезис, что squared-error metrics поощряют усреднение plausible fills, уже прямо сформулирован в этой работе.

### Kömürcü & Öksüz, 2026: Sharpening the Ensemble

Residual refiner обрабатывает ensemble двух моделей. На server validation SSIM повышается с 0.8555 до 0.8572, PSNR с 25.09 до 25.10 [D]. Это небольшой эффект. Paired significance анализ относится к отдельному внутреннему split, где base models ранее видели часть исходного training pool; авторы раскрывают это ограничение. Простое unsharp masking не воспроизводит gain. Описание L1 как conditional-mean minimizer в тексте математически неточно: стандартный L1 соответствует conditional median. Это не отменяет извлечённые экспериментальные числа. [Принятая публикация MICCAI](https://papers.miccai.org/miccai-2026-sat/BraTS_Inpainting_016.html).

### Cap, 2026: RESIN

Norm-free 3D SRResNet-подобная модель сохраняет входное разрешение. Patient-grouped folds и парные comparisons сравнивают её с настроенной U-Net. Три RESIN с flip TTA дают результат [E]. Авторы выделяют learning-rate annealing как сильную intervention; архитектура и training schedule оба важны. Ensemble composition выбирался на official validation. Работа не доказывает превосходства над winner 2025 и прямо это сообщает. [Статья](https://papers.miccai.org/miccai-2026-sat/BraTS_Inpainting_020.html).

### Guija-Valiente et al., 2026: RARF

Region-aware rectified flow восстанавливает известные voxels после каждого update. Для challenge применяется average 50 samples [F]. Авторы прямо связывают такое averaging с conditional mean, уменьшением MSE и возможной потерей fine structure. Это непосредственный предшественник обсуждения regression to the mean именно в BraTS inpainting. Число samples и отдельный submission checkpoint должны сопровождать результат.

### Persson, Bendazzoli & Moreno, 2026: Retrieval-Augmented Wavelet Diffusion

Prior строится усреднением ближайших real donor brains; он подаётся diffusion model. Используются дополнительные lesion-free HCP brains. Abstract сообщает internal held-out SSIM 0.879 [H]; это число не следует считать официальным test score команды, занявшей третье место. Научно важен явный донорский prior и анализ трудности интерьера больших voids. Для полноценного численного сравнения нужны таблицы на согласованном split и контроль дополнительного training data.

Дополнительный воспроизводимый entry: [CATCH, авторский репозиторий](https://github.com/simonwinther/brats2026). Для выбранного weighted-mixture mean-N=5 авторы сообщают organizer evaluation на 219 случаях: SSIM 0.772259, PSNR 20.886878, MSE 0.00982377. Репозиторий раскрывает selection и patient-overlap limitations; эти числа не делают entry SOTA и не заменяют данные реестра победителей.

## 6. Последние релевантные работы по метрикам

**Dohmen et al., Scientific Reports, 2025, Similarity and quality metrics for MR image-to-image translation.** Сравнение 11 reference и 12 non-reference метрик, 11 distortions, нормализаций и downstream segmentation. LPIPS, DISTS и стандартные метрики имеют разные профили чувствительности. Это MRI metric benchmark, но не проверка официальных inpainting entries и не reader validation. Предоставлены [полный текст](https://pmc.ncbi.nlm.nih.gov/articles/PMC11785996/) и [код](https://github.com/bayer-group/mr-image-metrics).

**Adamson et al., Magnetic Resonance in Medicine, 2025, Using deep feature distances for evaluating the perceptual quality of MR image reconstructions.** Сравнивает DFDs, включая MRI-specific и domain-adjacent features, с оценками пяти радиологов; DFDs и HFEN лучше согласуются с этими оценками, чем SSIM/PSNR в изученной reconstruction setting. Перенос на brain inpainting требует проверки; это не готовая замена challenge score. [Статья, код и данные](https://stanfordmimi.github.io/deep-feature-mr-recon/).

**Konz et al., Medical Image Analysis, 2026, Fréchet radiomic distance.** FRD сравнивает распределения медицинских изображений и сопровождается framework оценки метрик. Полезна для исследования реализма набора fills. Она не измеряет точность конкретного fill относительно его target и не является официальной BraTS metric. [Полный текст и ссылки на код](https://pmc.ncbi.nlm.nih.gov/articles/PMC13224392/).

## 7. Что можно утверждать по состоянию на дату обзора

1. Официальный benchmark продолжает использовать SSIM, PSNR и MSE/RMSE. В проверенных материалах нет принятой универсальной anatomical/perceptual замены.
2. Современные методы включают regression networks, diffusion, flow matching, ensembles и residual refinement. Нельзя описывать актуальное поле только через U-Net и ранние DDPM.
3. Averaging и blur уже прямо обсуждаются несколькими BraTS работами 2026. Сам этот тезис не следует представлять как впервые сформулированный нашим проектом.
4. Высокие distortion scores не являются отдельным доказательством patient-specific anatomy или экспертного качества. В то же время blur не означает автоматического проигрыша всех современных perceptual метрик.
5. Место команды, среднее SSIM и качество изображения отвечают разным вопросам. Доказательное сравнение требует совпадения cases, scorer, mask convention, aggregation и evaluation stage.

## 8. Дополнение после проверки всех camera-ready PDF

Ключевые уточнения к первоначальной выборке:

- В принятой версии Kömürcü & Öksüz уже есть десять training seeds: held-out SSIM имеет spread ±0.0051 и диапазон 0.8663–0.8780. Этот разброс превышает gain одного refiner над ensemble. Числа arXiv выше оставлены с указанием версии; для оценки устойчивости следует использовать camera-ready, раздел 5. В нём обсуждение MAE также не повторяет прежнюю формулировку L1 как mean minimizer.
- P3D-RAD, paper 014, уже измеряет SynthSeg Dice после inpainting: averaging 1→4 runs повышает mean Dice 0.845→0.860. Это непосредственный опубликованный downstream результат по рассматриваемой задаче. Он показывает изменение согласия с segmenter reference, а не независимое подтверждение правильности доопухолевой анатомии.
- Anatomy-Aware, paper 007, отдельно оценивает fidelity и segmentation consistency: на held-out fold SegDice растёт 64.22%→65.80% при малом изменении image-fidelity. Поэтому актуальная литература уже разделяет эти критерии.
- Enhanced U-Net, paper 002, сообщает SSIM 0.9065 на собственных 126 случаях. Сопоставление с числами прошлых winners из других публикаций не устанавливает победу на общей выборке.
- SSIM-Prioritized U-Net, paper 009, в abstract пишет online test, а Table 3 называет validation. В приложении stage отмечен как неоднозначный; дополнительные RMSE/MAE строки не используются для межметодного рейтинга без прояснения их происхождения.

Источники: camera-ready PDF 016 (разделы 4–5), 014 (Table 1), 007 (Tables 5–6), 002 (Table 1), 009 (Table 3); прямые ссылки приведены в приложении. Эти уточнения заменяют более ранние выводы, основанные только на arXiv/abstract.

## 9. Неподтверждённые элементы снимка

- Не получена полная официальная численная hidden-test таблица 2026 со всеми командами и размерами test subsets. Поэтому здесь нет выдуманного единого численного SOTA leaderboard.
- Не проверено тождество всех paper checkpoints контейнерам финальных submissions.
- Не выполнено воспроизведение внешних моделей или независимое получение их server scores.
- Все 16 PDF каталога получены и проверены в части методов, результатов и границ сравнения. Это не аудит каждой формулы, библиографической ссылки, строки кода или визуальной оценки.
- Повторный поиск по официальному сайту, GitHub challenge и именам top teams не дал полной публичной numerical hidden-test таблицы 2026. Challenge Portal отдаёт страницу, требующую JavaScript; публичные validation результаты доступны в статьях. Пробел относится к финальным numerical test scores, а не к наличию публикаций или рангов в Orchestrator.
- Наши QTIM/synthetic/iris результаты не включены в внешнюю SOTA-таблицу: это другой набор данных и протокол.

Следующее обновление этого документа должно добавлять проверенные official test rows или новые первичные публикации с явными split/version labels. Разработку новой метрики и расширение Identity Ladder этот снимок не запускает.
