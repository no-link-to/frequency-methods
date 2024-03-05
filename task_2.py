import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


class FourierComplex:
    __t_list = []
    __ft_list = []
    __coefficients_c = []
    __complex_sum = []
    __period = 16
    __N = 0

    def __init__(self, N):
        self.__N = N

    def __func(self, t):
        if -2 <= t < 2:
            return 5 + 2.5 * t * 1j
        elif 2 <= t < 6:
            return 10 - 2.5 * t + 5 * 1j
        elif 6 <= t < 10:
            return -5 + (20 - 2.5 * t) * 1j
        elif 10 <= t < 14:
            return -30 + 2.5 * t - 5 * 1j
        return 0

    def __calculate_t(self):
        self.__t_list = np.linspace(-2, 14, 1000)

    def __calculate_ft(self):
        self.__ft_list = [self.__func(item) for item in self.__t_list]

    def __calculate_coefficients(self):
        result = []
        for n in range(-self.__N, self.__N + 1):
            tmp = 1 / 16 * integrate.quad(lambda t: self.__func(t) * np.exp(-1j * np.pi * n * t / 8), -2, 14)[0]
            result.append(round(tmp, 3))
        self.__coefficients_c = result

    def __calculate_complex_sum(self):
        result = []
        for t in self.__t_list:
            tmp_c = [*self.__coefficients_c]
            tmp_result = 0
            for i in range(-self.__N, self.__N + 1):
                tmp_result += tmp_c[i + self.__N] * np.exp(1j * 2 * np.pi * i * t / self.__period)
            result.append(tmp_result)
        self.__complex_sum = result

    def __draw(self):
        plt.figure(figsize=(10, 6))
        plt.plot([t.real for t in self.__ft_list if t],
                 [t.imag for t in self.__ft_list if t])
        plt.plot([item.real for item in self.__complex_sum if item],
                 [item.imag for item in self.__complex_sum if item])
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.legend()
        plt.grid(True)
        plt.show()

    def __draw_re_parts(self):
        plt.figure(figsize=(10, 6))
        real_ft = [t.real for t in self.__ft_list if t]
        real_sum = [item.real for item in self.__complex_sum]
        plt.plot(self.__t_list[:len(real_ft)], real_ft)
        plt.plot(self.__t_list[:len(real_sum)], real_sum)
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.legend()
        plt.grid(True)
        plt.show()

    def __draw_im_parts(self):
        plt.figure(figsize=(10, 6))
        imag_ft = [t.imag for t in self.__ft_list if t]
        imag_sum = [item.imag for item in self.__complex_sum]
        plt.plot(self.__t_list[:len(imag_ft)], imag_ft)
        plt.plot(self.__t_list[:len(imag_sum)], imag_sum)
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.legend()
        plt.grid(True)
        plt.show()

    def __check_parseval(self):
        func = (1 / 16) * (integrate.quad(lambda x: self.__func(x).real, -2, 14)[0] +
                           integrate.quad(lambda x: self.__func(x).imag, -2, 14)[0] * 1j)

        parseval_complex = sum(c.real**2 + c.imag**2 for c in self.__coefficients_c)

        print(f"Сумма (комплексный случай) {parseval_complex} квадрат нормы {func} равны "
              f"с погрешностью 0.1: {np.isclose(parseval_complex, func, atol=(1e-1 + 1j))}")

    def __display_coefficients(self):
        print(f"Cn: {self.__coefficients_c}")

    def run(self):
        self.__calculate_t()
        self.__calculate_ft()
        self.__calculate_coefficients()
        self.__calculate_complex_sum()
        self.__display_coefficients()
        self.__check_parseval()
        self.__draw()
        self.__draw_re_parts()
        self.__draw_im_parts()


if __name__ == "__main__":
    # example = FourierComplex(1)
    # example = FourierComplex(2)
    # example = FourierComplex(3)
    example = FourierComplex(10)
    example.run()
