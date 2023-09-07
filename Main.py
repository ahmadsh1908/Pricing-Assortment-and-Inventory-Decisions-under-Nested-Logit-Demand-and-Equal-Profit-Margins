# import pandas as pd
import numpy as np
from scipy.special import lambertw
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.special import logsumexp


def rho():  # numerator on equation 2
    return sum([nestutility(n)**(mu[2]/mu[1]) for n in Nest])


def nestutility(n):  # sumation of all nests denominator on equation 2 excluding v0
    return sum([netu(i) for i in Nest[n]])


def zeta(x):  # zeta represents this symbol on the paper ζli
    # return ζli
    return np.sqrt(netu(x)*(nestutility(x % 10)+0.00001)**(((mu[2]/mu[1]) - 1)))


def p(m):
    M = np.exp(-m/mu[1])  # reprsents numerator on the eq
    return M/(v0+rho()*M)  # return p(N,m)


def profitriskless(m):  # equation 11 or first term from equation 8
    return m*L*p(m)*rho()


def theta(m):  # equation for θ
    def Y(n): return sum([(c[l]/(c[l]+m))*zeta(l) for l in Nest[n]])
    return sum([Y(n) for n in Nest])  # returning  θ(N,m)


def thetalowerbound(m):
    def E(n): return max([(c[l]/(c[l]+m))
                          for l in Nest[n]])  # get higher cost on nest i
    def Y(n): return sum([E(n)*zeta(l)
                          for l in Nest[n]])  # formula for lambda on each nest
    return sum([Y(n) for n in Nest])


def thetaupperbound(m):  # used in equation 17
    def E(n): return min([(c[l]/(c[l]+m))
                          for l in Nest[n]])  # get lower cost on nest i
    def Y(n): return sum([E(n)*zeta(l)
                          for l in Nest[n]])  # formula for lambda on each nest
    return sum([Y(n) for n in Nest])


def Cinv(m):
    # formula for overage cost or 2nd term on equation 8
    return 1.66*m*np.sqrt(L*p(m))*theta(m)


def profit(m):  # returns value for equation 8
    return profitriskless(m)-Cinv(m)


def profitupper(m):  # equation 17 where overage cost uses theta upper bound
    return profitriskless(m)-Cinvupper(m)


def profitlower(m):  # equation 18 where overage cost uses theta upper bound
    return profitriskless(m)-Cinvlower(m)


def Cinvupper(m):  # equation 13 Cost with theta upper bound
    return 1.66*m*np.sqrt(L*p(m))*thetaupperbound(m)


def Cinvlower(m):  # equation 13 Cost with theta lower  bound
    return 1.66*m*np.sqrt(L*p(m))*thetalowerbound(m)


def profitexact(m):  # equation 7
    return profitriskless(m)-Cinvexact(m)


def Cinvexact(m):  # 2nd term on equation 7
    def Y(n): return sum(
        [(m+c[l])*zeta(l)*norm.pdf(norm.ppf(1-c[l]/(c[l]+m))) for l in Nest[n]])
    return np.sqrt(L*p(m))*sum([Y(n) for n in Nest])


def qnest(n, p):  # equation 2
    def Y(n): return sum([np.exp((alpha[l]-p[l]-A[l])/mu[2])
                          for l in Nest[n]])**(mu[2]/mu[1])
    return Y(n)/(v0+sum([Y(k) for k in Nest]))


def qitem(l, p):  # equation 1
    n = l % 10
    return np.exp((alpha[l]-p[l]-A[l])/mu[2]) / sum([np.exp((alpha[i]-p[i]-A[i])/mu[2]) for i in Nest[n]])


def qli(l, p):
    n = l % 10
    return qitem(l, p)*qnest(n, p)  # returns equation 4


def profitp(p):  # Optimizing profit over price
    try:
        def Y(l): return (p[l]-c[l])*L*qli(l, p)-p[l] * \
            1.66*(c[l]/p[l])*(1-c[l]/p[l])*np.sqrt(L*qli(l, p))
        return sum([Y(l) for l in p])  # returns equation [8 or 7]
    except:
        return -100


def profitparray(parray):  # returns profit values for different assormtnets
    keys = list(c.keys())
    pdict = {keys[i]: parray[i] for i in range(len(keys))}
    return profitp(pdict)


def netu(i):  # numerator on equation 1
    return np.exp((alpha[i]-c[i]-A[i])/mu[2])


def profitupperlowervsm():  # profit upper and lower bounds
    global profittlower, profittupper, profitt, maxgap, risklessprofit, profittexact
    profittlower = [profitlower(m) for m in np.arange(0, 30.5, 0.5)]
    profittupper = [profitupper(m) for m in np.arange(0, 30.5, 0.5)]
    profitt = [profit(m) for m in np.arange(0, 30.5, 0.5)]
    maxgap = [(profitupper(m)-profitlower(m))/profitlower(m)
              for m in np.arange(0.5, 30.5, 0.5)]
    risklessprofit = [profitriskless(m) for m in np.arange(0, 30.5, 0.5)]


def optimalm():  # optimal m values for different profit functions (riskless, upper, lower, exact)
    global m0numerical, mnumerical, profitt, m0closedform, mnumericalexact
    m0numerical = optimize.fminbound(lambda m: -profitriskless(m), 0, 1000)
    m0closedform = np.real(mu[1]*(1+lambertw(np.exp(-1)*rho()/v0)))
    mnumerical = optimize.fminbound(
        lambda m: -profit(m), 0, brentq(profit, 1, 1000))
    mnumericalexact = optimize.fminbound(lambda m: -profitexact(m), 0, 15)


def assortmentassesment(N):
    global A
    global Assortmentperformance
    mnumerical = optimize.fminbound(
        lambda m: -profit(m), 0,  brentq(profit, 1, 1000))
    Assortmentperformance[tuple(A.values())] = (mnumerical, profit(mnumerical))

    i = 0
    for i1 in Nest[N]:
        A[Nest[N][-i-1]] = 1000
        mnumerical = optimize.fminbound(lambda m: -profit(m), 0,  20)
        Assortmentperformance[tuple(A.values())] = (
            mnumerical, profit(mnumerical))
        i += 1

    i = 0
    for i1 in Nest[N]:
        A[Nest[N][-i-1]] = 0
        i += 1


def assortmentassesment2nests(N1, N2):  # profit assortment between 2 nests
    global A
    global Assortmentperformance
    i = 0
    assortmentassesment(N1)
    for i1 in Nest[N2]:
        A[Nest[N2][-i-1]] = 1000
        assortmentassesment(N1)
        i += 1

    i = 0
    for i1 in Nest[N2]:
        A[Nest[N2][-i-1]] = 0
        i += 1


def assortmentassesmentnests():  # profit assortment between all nests
    global A
    global Assortmentperformance
    assortmentassesment2nests(1, 2)

    i = 0
    for i1 in Nest[3]:
        A[Nest[3][-i-1]] = 1000
        assortmentassesment2nests(1, 2)
        i += 1

    i = 0
    for i1 in Nest[3]:
        A[Nest[3][-i-1]] = 1000
        i += 1


def lambdamin(vo):
    a = 1.66*1.66

    return (a*(vo+rho()))*((sum(zeta(l) for n in Nest for l in Nest[n]))**2)/rho()**2


def main():
    # A is variable that tells if the item is in an assortment
    global Nest, c, alpha, mu, L, v0, A, Assortmentperformance

    Nest = {1: [], 2: [], 3: []}
    c = {11: 4, 21: 5, 31: 6, 41: 7, 12: 8, 22: 9,
         32: 10, 42: 11, 13: 12, 23: 13, 33: 14, 43: 15}

    A = {}
    Assortmentperformance = {}
    alpha = {11: 9, 21: 8, 31: 12, 41: 9, 12: 14, 22: 12,
             32: 14, 42: 12, 13: 17, 23: 17, 33: 18, 43: 22}

    mu = {1: 2, 2: 1.2}
    v0 = 1
    L = 100

    for items in c:
        k = items % 10
        Nest[k].append(items)
        A[items] = 0
        # netu[items]=np.exp((alpha[items]-c[items])/mu[2])

    for n in Nest:
        Nest[n] = sorted(Nest[n], key=lambda s: alpha[s]-c[s], reverse=True)

    assortmentassesmentnests()
    global resultsassortment
    resultsassortment = {}
    for i in Assortmentperformance:
        resultsassortment[i] = Assortmentperformance[i][1]
    global Keymax
    Keymax = max(resultsassortment, key=resultsassortment.get)
    print(Keymax)

    profitupperlowervsm()
    optimalm()

    # Optimizing profit over price vector (remove comment to run)
    # global poptimal
    # global p0
    # y=c.values()
    # x=alpha.values()
    # zip_object = zip(x, y)
    # p0=list(alpha.values())
    # # for list1_i, list2_i in zip_object:
    # #     p0.append(list1_i-list2_i+5)
    # global p0tuple
    # p0tuple=tuple((p0[i],p0[i]+200) for i in range(len(p0)))

    # # res = minimize(lambda pi: -profitparray(pi), p0, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
    # # res = minimize(lambda pi: -profitparray(pi), p0, method='SLSQP',bounds=p0tuple,options={'disp': True})
    # res = minimize(lambda pi: -profitparray(pi), p0, method='BFGS', jac=p0tuple, options={'disp': True})
    # poptimal=res.x
    # print(profitparray(poptimal))


if __name__ == '__main__':
    main()
