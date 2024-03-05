import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


class Fourier:
    __N = 0
    __period_start = 0
    __period_end = 0
    __period = 0
    __t_list = []
    __ft_list = []
    __coefficients_a = []
    __coefficients_b = []
    __coefficients_c = []
    __series = []
    __func_type = 1

    def __init__(self, N, period_start, period_end, func_type):
        self.__N = N
        self.__period_start = period_start
        self.__period_end = period_end
        self.__period = period_end - period_start
        self.__func_type = func_type

    def __func_square(self, value):
        return 4 if value % self.__period >= 5 else 5

    def __func_even(self, value):
        return np.cos(3 * value) + (np.sin(5 * value))**2

    def __func_odd(self, value):
        return np.sin(6 * value) + (np.sin(8 * value))**3

    def __func_other(self, value):
        return np.sin(2 * value) + (np.sin(5 * value))**2

    def __get_func(self):
        return [self.__func_square, self.__func_even, self.__func_odd, self.__func_other][self.__func_type - 1]

    def __calculate_t(self):
        self.__t_list = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

    def __calculate_ft(self):
        self.__ft_list = [self.__get_func()(item) for item in self.__t_list]

    def __calculate_coefficients_square(self):
        a_0 = (2 / self.__period) * (integrate.quad(lambda t: 4, self.__period_start, self.__period)[0] +
                                     integrate.quad(lambda t: 5, self.__period, self.__period_end)[0])
        result_a = [round(a_0, 3)]
        result_b = []
        result_c = [round(a_0, 3) / 2]
        for i in range(1, self.__N + 1):
            a_n = (2 / self.__period) * (integrate.quad(lambda t: 4 * np.cos(2 * np.pi * i * t / self.__period),
                                                        self.__period_start, self.__period)[0] +
                                         integrate.quad(lambda t: 5 * np.cos(2 * np.pi * i * t / self.__period),
                                                        self.__period, self.__period_end)[0])
            b_n = (2 / self.__period) * (integrate.quad(lambda t: 4 * np.sin(2 * np.pi * i * t / self.__period),
                                                        self.__period_start, self.__period)[0] +
                                         integrate.quad(lambda t: 5 * np.sin(2 * np.pi * i * t / self.__period),
                                                        self.__period, self.__period_end)[0])
            result_a.append(round(a_n, 3))
            result_b.append(round(b_n, 3))
            result_c.insert(0, round(a_n, 3) / 2 + round(b_n, 3) / 2 * 1j)
            result_c.append(round(a_n, 3) / 2 - round(b_n, 3) / 2 * 1j)
        self.__coefficients_a = result_a
        self.__coefficients_b = result_b
        self.__coefficients_c = result_c

    def __calculate_coefficients(self):
        a_0 = (2 / self.__period) * integrate.quad(lambda t: self.__get_func()(t), self.__period_start, self.__period_end)[0]
        result_a = [round(a_0, 3)]
        result_b = []
        result_c = [round(a_0, 3) / 2]
        for i in range(1, self.__N + 1):
            a_n = (2 / self.__period) * integrate.quad(lambda t: self.__get_func()(t) * np.cos(2 * np.pi * i * t / self.__period),
                                                       self.__period_start, self.__period_end)[0]
            b_n = (2 / self.__period) * integrate.quad(lambda t: self.__get_func()(t) * np.sin(2 * np.pi * i * t / self.__period),
                                                       self.__period_start, self.__period_end)[0]
            result_a.append(round(a_n, 3))
            result_b.append(round(b_n, 3))
            result_c.insert(0, round(a_n, 3) / 2 + round(b_n, 3) / 2 * 1j)
            result_c.append(round(a_n, 3) / 2 - round(b_n, 3) / 2 * 1j)
        self.__coefficients_a = result_a
        self.__coefficients_b = result_b
        self.__coefficients_c = result_c

    def __calculate_sum(self, t):
        tmp_a = [*self.__coefficients_a]
        tmp_b = [*self.__coefficients_b]
        result = tmp_a.pop(0) / 2
        for i in range(1, self.__N + 1):
            result += tmp_a[i - 1] * np.cos(2 * np.pi * i * t / self.__period)
            result += tmp_b[i - 1] * np.sin(2 * np.pi * i * t / self.__period)
        return result

    def __calculate_complex_sum(self, t):
        tmp_c = [*self.__coefficients_c]
        result = 0
        for i in range(-self.__N, self.__N + 1):
            result += tmp_c[i + self.__N] * np.exp(1j * 2 * np.pi * i * t / self.__period)
        return result

    def __check_parseval(self):
        parseval = ((self.__coefficients_a[0]**2) / 2 +
                    sum(a**2 for a in self.__coefficients_a[1:]) +
                    sum(b ** 2 for b in self.__coefficients_b))
        func = (2 / self.__period) * integrate.quad(lambda t: self.__get_func()(t) * self.__get_func()(t),
                                                    self.__period_start, self.__period_end)[0]
        parseval_complex = 2 * sum(c.real**2 + c.imag**2 for c in self.__coefficients_c)
        print(f"Сумма {parseval} квадрат нормы {func} равны "
              f"с погрешностью 0.1: {np.isclose(parseval, func, atol=1e-1)}")
        print(f"Сумма (комплексный случай) {parseval_complex} квадрат нормы {func} равны "
              f"с погрешностью 0.1: {np.isclose(parseval_complex, func, atol=1e-1)}")

    def __draw(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__t_list, self.__ft_list, label='f(t)')
        plt.plot(self.__t_list, [self.__calculate_complex_sum(t) for t in self.__t_list], label='Gn(t)')
        plt.plot(self.__t_list, [self.__calculate_sum(t) for t in self.__t_list], label='Fn(t)')
        plt.xlabel('t')
        plt.ylabel('f(t)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def __display_coefficients(self):
        print(f"An: {self.__coefficients_a}")
        print(f"Bn: {self.__coefficients_b}")
        print(f"Cn: {self.__coefficients_c}")
        self.__check_parseval()

    def run(self):
        self.__calculate_t()
        self.__calculate_ft()
        self.__calculate_coefficients_square() if self.__func_type == 1 else self.__calculate_coefficients()
        self.__display_coefficients()
        self.__draw()


if __name__ == '__main__':
    # example = Fourier(3, 5, 15, 1)
    # example = Fourier(10, 5, 15, 1)
    # example = Fourier(20, 5, 15, 1)
    # example = Fourier(30, 5, 15, 1)
    # example = Fourier(40, 5, 15, 1)

    # example = Fourier(3, -np.pi, np.pi, 2)
    # example = Fourier(10, -np.pi, np.pi, 2)
    # example = Fourier(20, -np.pi, np.pi, 2)
    example = Fourier(30, -np.pi, np.pi, 2)
    # example = Fourier(40, -np.pi, np.pi, 2)

    # example = Fourier(3, 0, np.pi, 3)
    # example = Fourier(10, 0, np.pi, 3)
    # example = Fourier(20, 0, np.pi, 3)
    # example = Fourier(30, 0, np.pi, 3)
    # example = Fourier(40, 0, np.pi, 3)
    #
    # example = Fourier(3, 0, np.pi, 4)
    # example = Fourier(10, 0, np.pi, 4)
    # example = Fourier(20, 0, np.pi, 4)
    # example = Fourier(30, 0, np.pi, 4)
    # example = Fourier(40, 0, np.pi, 4)

    example.run()
