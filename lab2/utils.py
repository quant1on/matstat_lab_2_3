import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from scipy.stats import variation
from statistics import mode


# Генерация выборки распределения Бернулли
def bernoulli(n, p, size=1000):
    """
    :param n: Количество испытаний в каждом эксперименте
    :param p: Вероятность успеха при каждом испытании
    :param size: Размер выборки
    :return: Массив сгенерированных данных
    """
    distribution = np.random.binomial(n=n, p=p, size=size)
    return distribution


# Генерация выборки эскпоненциального распределения
def exponential(scale, size=1000):
    """
    :param scale: Масштаб (1/lambda)
    :param size: Размер выборки
    :return: Массив сгенерированных данных
    """
    distribution = np.random.exponential(scale=scale, size=size)
    return distribution


# Вычисление квартилей
def quantiles(part, data):
    """
    :param part: Список квантилей, которые хотим рассчитать
    :param data: Данные на вход
    :return: Массив значений полученных квантилей
    """
    q = np.quantile(data, part)
    return q


# Вычисление и вывод мер центральной тенденции
def central_stats(data, dist_name):
    """

    :param data: Данные на вход
    :param dist_name: Название распределения для вывода в print
    """
    print(f"\nМеры центральной тенденции для распределения: {dist_name}:")

    # Квартили
    q1, q2, q3 = np.quantile(data, [0.25, 0.5, 0.75])

    # Среднее значение
    mean = np.mean(data)

    # Медиана
    median = np.median(data)

    # Мода
    moda = mode(data)


    # Вывод результатов
    print(f"• Первый квартиль (Q1) = {q1:.4f}")
    print(f"• Медиана (Q2) = {q2:.4f}")
    print(f"• Третий квартиль (Q3) = {q3:.4f}")
    print(f"• Среднее значение = {mean:.4f}")
    print(f"• Медиана = {median:.4f}")
    print(f"• Мода = {moda}" + (f" (встречается раз: {moda} )" if moda else ""))


#Вычисление и вывод мер вариабельности
def variabiliyy_stats(data, dist_name):
    """

    :param data: Данные на вход
    :param dist_name: Название распределения для вывода в print
    """
    print(f"\nМеры вариабельности для следующего распределения: {dist_name}:")

    # Размах
    data_range = np.max(data) - np.min(data)

    # Интерквартильный размах (IQR)
    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3 - q1

    # Дисперсия (несмещенная)
    variance = np.var(data, ddof=1)

    # Стандартное отклонение (несмещенное)
    std_dev = np.std(data, ddof=1)

    # Коэффициент вариации (в процентах)
    cv = variation(data) * 100

    # Среднее абсолютное отклонение
    mad = np.mean(np.abs(data - np.mean(data)))

    # Вывод результатов
    print(f"• Размах = {data_range:.4f}")
    print(f"• Интерквартильный размах (IQR) = {iqr:.4f}")
    print(f"• Дисперсия = {variance:.4f}")
    print(f"• Стандартное отклонение = {std_dev:.4f}")
    print(f"• Коэффициент вариации (CV) = {cv:.4f}%")
    print(f"• Среднее абсолютное отклонение (MAD) = {mad:.4f}")

# Вычисление и вывод мер форм распределения
def distribution_shape_stats(data, dist_name):
    """

    :param data: Данные на вход
    :param dist_name:
    :return:
    """
    print(f"\nМеры формы распределения {dist_name}:")

    # Коэффициент асимметрии
    skewness = stats.skew(data)
    print(f"• Коэффициент асимметрии = {skewness:.4f}")

    # Коэффициент эксцесса
    kurtosis = stats.kurtosis(data)
    print(f"• Коэффициент эксцесса = {kurtosis:.4f}")

    # Начальные моменты (1-5)
    print("\nНачальные моменты:")
    for k in range(1, 6):
        moment = np.mean(data ** k)
        print(f"• M_{k} = {moment:.4f}")

    # Центральные моменты (1-5)
    print("\nЦентральные моменты:")
    for k in range(1, 6):
        moment = stats.moment(data, moment=k)
        print(f"• μ_{k} = {moment:.4f}")

# Построение графиков распределения
def plot_distributions(data, dist_name, dist_type, params):
    """

    :param data: Данные на вход
    :param dist_name: Название распределения для вывода в print
    :param dist_type: Тип распределения (непрерывный или дискретный соответственно)
    :param params: Параметры для распределения
    """
    plt.figure(figsize=(15, 10))

    # Для непрерывных распределений
    if dist_type == "continuous":
        # Гистограмма с разными бинами
        bin_methods = [
            ('auto', 'Автоматический выбор'),
            ('sturges', 'Формула Старджеса'),
            ('sqrt', 'Квадратный корень'),
            (20, 'Ручной выбор (20 бинов)')
        ]

        for i, (bins, title) in enumerate(bin_methods, 1):
            plt.subplot(2, 2, i)
            counts, bins, _ = plt.hist(data, bins=bins, density=True, alpha=0.6, edgecolor='black')

            # Теоретическая PDF
            x = np.linspace(np.min(data), np.max(data), 1000)
            if dist_name == "Экспоненциальное":
                pdf = stats.expon(scale=params['scale']).pdf(x)
            plt.plot(x, pdf, 'r-', linewidth=2)

            plt.title(f"{dist_name}\nГистограмма ({title})")
            plt.xlabel('Значение')
            plt.ylabel('Плотность вероятности')

    # Для дискретных распределений
    elif dist_type == "discrete":
        # Многоугольник вероятностей
        unique, counts = np.unique(data, return_counts=True)
        probs = counts / len(data)
        plt.subplot(2, 1, 1)
        plt.stem(unique, probs)

        # Теоретическая PMF
        if dist_name == "Биномиальное":
            x = np.arange(0, params['n'] + 1)
            pmf = stats.binom(n=params['n'], p=params['p']).pmf(x)
            plt.plot(x, pmf, 'ro', markersize=5)

        plt.title(f"{dist_name}\nЭмпирическая функция вероятности")
        plt.xlabel('Значение')
        plt.ylabel('Вероятность')

    # ECDF и теоретическая CDF
    plt.subplot(2, 2, 3) if dist_type == "continuous" else plt.subplot(2, 1, 2)
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.step(x, y, where='post', label='eCDF')

    # Теоретическая CDF
    if dist_name == "Биномиальное":
        x_theor = np.arange(0, params['n'] + 1)
        cdf = stats.binom(n=params['n'], p=params['p']).cdf(x_theor)
        plt.step(x_theor, cdf, 'r-', where='post', label='Теоретическая CDF')
    elif dist_name == "Экспоненциальное":
        x_theor = np.linspace(0, np.max(data), 1000)
        cdf = stats.expon(scale=params['scale']).cdf(x_theor)
        plt.plot(x_theor, cdf, 'r-', label='Теоретическая CDF')

    plt.title('Эмпирическая и теоретическая CDF')
    plt.xlabel('Значение')
    plt.ylabel('Вероятность')
    plt.legend()

    # Boxplot
    plt.subplot(2, 2, 4) if dist_type == "continuous" else None
    if dist_type == "continuous":
        plt.boxplot(data, vert=False, patch_artist=True)
        plt.title('Boxplot с выбросами')
        plt.xlabel('Значение')

    plt.tight_layout()
    plt.show()


# Добавление выбросов в данные
def add_outliers(data, outlier_percent, dist_type, params):
    """

    :param data: Изначальные данные на вход
    :param outlier_percent: Процент выбросов
    :param dist_type: Тип распределения
    :param params: Параметры распределения
    :return: Массив данных с выбросами
    """
    n = len(data)
    num_outliers = int(n * outlier_percent / 100)

    if dist_type == "exponential":
        threshold = stats.expon.ppf(0.999, scale=params['scale'])
        outliers = stats.expon.rvs(scale=params['scale'], size=num_outliers) + threshold
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

    # Замена случайных элементы
    indices = np.random.choice(n, num_outliers, replace=False)
    data_with_outliers = data.copy()
    data_with_outliers[indices] = outliers

    return data_with_outliers

# Функция для анализа устойчивости характеристик к выбросам
def robustness_analysis(data, dist_name, dist_type, params):
    """

    :param data: Данные на вход
    :param dist_name: Название распределения
    :param dist_type: Тип распределения
    :param params: Параметры распределения
    :return:
    """
    # Исходные статистики
    print("\n" + "=" * 50)
    print("Исходные статистики:")
    central_stats(data, dist_name)
    variabiliyy_stats(data, dist_name)

    # 5% выбросов
    data_5perc = add_outliers(data, 5.0, dist_type, params)

    print("\n" + "=" * 50)
    print("Статистики с 5% выбросов:")
    central_stats(data_5perc, f"{dist_name} с выбросами")
    variabiliyy_stats(data_5perc, f"{dist_name} с выбросами")

    # Графики изменения мер вариабельности
    percentages = np.linspace(0, 5, 11)
    metrics = {
        'Размах': [],
        'IQR': [],
        'Дисперсия': [],
        'Стандартное отклонение': [],
        'CV (%)': [],
        'MAD': []
    }

    for p in percentages:
        data_p = add_outliers(data, float(p), dist_type, params)
        metrics['Размах'].append(np.max(data_p) - np.min(data_p))
        q1, q3 = np.quantile(data_p, [0.25, 0.75])
        metrics['IQR'].append(float(q3 - q1))
        metrics['Дисперсия'].append(float(np.var(data_p, ddof=1)))
        metrics['Стандартное отклонение'].append(float(np.std(data_p, ddof=1)))
        metrics['CV (%)'].append(float(variation(data_p) * 100))
        metrics['MAD'].append(float(np.mean(np.abs(data_p - np.mean(data_p)))))

    plt.figure(figsize=(15, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (metric, values) in enumerate(metrics.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(percentages, values,
                 color=colors[i - 1],
                 marker='o',
                 linestyle='--',
                 linewidth=2,
                 markersize=8)
        plt.title(metric, fontsize=12)
        plt.xlabel('Процент выбросов (%)', fontsize=10)
        plt.ylabel('Значение', fontsize=10)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f"Изменение мер вариабельности для {dist_name}", y=1.02, fontsize=14)
    plt.show()




