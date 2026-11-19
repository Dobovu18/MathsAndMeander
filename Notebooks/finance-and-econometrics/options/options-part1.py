import marimo

__generated_with = "0.25.2"
app = marimo.App(css_file="../../_theme/blackboard.css")

with app.setup:
    # Import libraries and set up the notebook in the initialisation

    # Imports for notebook set-up, plots and themes:
    import marimo as mo
    import matplotlib.pyplot as plt

    from mathsandmeander.common import bib, theme, xref

    theme.apply(usetex=True)

    # Imports for numerical and data analysis:
    import numpy as np
    from scipy.optimize import root_scalar


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Financial Theory – Options (Part I):
    By Donnell Obovu

    ### I. Preface:

    This is Part I of a four-part series of interactive notebooks whose purpose is to introduce financial options.
    Each part works through a different aspect of option theory:
    * Part I: Introduction to Options
    * Part II: Working with Data on Options
    * Part III: Methods for Pricing Options
    * Part IV: Strategies for Trading Options

    The power of using these notebooks is their interactive capability, which makes the theory more tangible, and allows for the data used in case-studies to be easily visualised and clearly presented. I therefore encourage readers to interact with the widgets present within the notebook.

    The intended audience is those with a mathematical background, but no prior knowledge of finance. The prerequisite knowledge required to understand the notebook will be stated towards the start of each notebook.

    ### II. Prerequisites:
    For this notebook, I recommend that readers have an understanding of:
    - Real Analysis and Calculus

    ### III. Contents:
    * [1. Financial Foundations and Introducing Options](#sec-foundations)
        * [1.1. Finance Basics](#sec-finance-basics)
        * [1.2. Derivatives](#sec-derivatives)
        * [1.3. What Are Options?](#sec-what-are-options)
        * [1.4. Example – The European Call Option](#sec-european-call)
        * [1.5. Example – The European Put Option](#sec-european-put)
        * [1.6. Intrinsic Value and Moneyness](#sec-moneyness)
        * [1.7. The Use of Options](#sec-use-of-options)
        * [1.8. Common Assumptions in Mathematical Finance](#sec-assumptions)
        * [1.9. Strategies Involving Options](#sec-strategies)
        * [1.10. The Forward Price](#sec-forward-price)
    * [2. Properties of the Prices of Options](#sec-price-properties)
        * [2.1. Bounds on the Price of Options](#sec-price-bounds)
        * [2.2. Put-Call Parity](#sec-put-call-parity)
        * [2.3. The Effect of Dividends](#sec-dividends)
        * [2.4. Early Exercise of American Options](#sec-early-exercise)

    ### IV. Notation:
    Prices and payments are in units of currency, so formulae carry no currency symbol; worked examples and plots use US dollars (\$). Time is measured in years. Unless stated otherwise:

    | Symbol | Meaning |
    | ------ | ------- |
    | $t$, $T$ | the current time; the time of maturity |
    | $r$ | the risk-free interest rate per annum (continuously compounded) |
    | $S(t)$, $S$ | the price of the underlying asset at time $t$ (so $S(0)$ today and $S(T)$ at maturity); $S$ when the time is clear from the context |
    | $K$ | the strike price of an option, or the delivery price of a forward |
    | $C(t)$, $P(t)$ | the price (premium) at time $t$ of a European call, put |
    | $C_{E}$, $C_{A}$, $P_{E}$, $P_{A}$ | the price of a European call, American call, European put, American put, written $C_{E}(t, T, S, K)$ when the time, expiry, asset price or strike varies (Section 2) |
    | $c$, $F$ | the coupon and face value of a bond |
    | $Z(t, T, r)$ | the discount factor: the value at time $t$ of $1$ paid at time $T$ (Section 1.1.1) |
    | $B(t, T, r)$ | the compounding factor: the amount owed at time $T$ on $1$ borrowed at time $t$ (Section 1.1.1) |
    | $V(t)$ | an amount of money at time $t$ (Section 1.1.1), or the value of a bond at time $t$ (Section 1.1.3) |
    | $D$ | the present value of the dividends paid during the lifetime of an option (Section 2.3); $D(t)$ is the present value at time $t$ of the dividends paid between $t$ and the maturity of a forward (Section 1.10) |
    | $f(t, T, S, K)$ | the value at time $t$ of a long forward contract with maturity $T$ and delivery price $K$ (Section 1.10) |
    | $K_{*}(t, T, S)$ | the forward price at time $t$ for delivery at time $T$ (Section 1.10) |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-foundations"></span>
    ## 1. Financial Foundations and Introducing Options:

    <span id="sec-finance-basics"></span>
    ### 1.1. Finance Basics:

    #### 1.1.1. Interest, Compounding and Discounting:

    Money today is better than money tomorrow, so lenders of money wish to be compensated by borrowers for not having access to their money over a given period of time [[Joshi 2003](#ref-MJ03)]. Lending money also exposes the lender to the risk of the borrower not paying it back. These are the ideas behind interest.

    Interest is quantified by the _interest rate_, which is the percentage of the borrowed money the borrower must pay the lender per unit of time (typically per annum). Therefore, if $V(t)$ is the amount owed at time $t$, $r$ is the interest rate at time $t$, and time $\mathrm{d}t$ has elapsed,

    <span id="eq-interest-ode"></span>
    $$\frac{\mathrm{d}V}{V} = r\, \mathrm{d}t \tag{1.1.1}$$

    where $\mathrm{d}V = V(t+\mathrm{d}t) - V(t)$ [[Wilmott, Howison, Dewynne 1995](#ref-WHD95)].

    The equation above determines the amount owed when provided with a known (deterministic) interest rate.

    In the cases where the interest rate $r$ is a constant rate per annum:
    * If we compound $n$ times per annum, the interest is added in discrete steps of length $\frac{1}{n}$ (so $\mathrm{d}t$ is replaced by the finite step $\frac{1}{n}$). If the initial amount owed is $V(0)$, then, for $tn\in\N$,

    <span id="eq-discrete-comp"></span>
    $$V(t) = V(0)\left(1 + \frac{r}{n}\right)^{tn} \tag{1.1.2}$$
    is owed after time $t$ has elapsed.
    * If we compound continuously, then solving the differential equation gives

    <span id="eq-cont-comp"></span>
    $$V(t) = V(0)e^{rt} \tag{1.1.3}$$

    A related question is: _how much would I pay now to receive a (guaranteed) payment of $V(T)$ at time $T$ in the future?_

    Suppose the interest rate $r$ is known at every time up until time $T$. We can answer this by solving [(1.1.1)](#eq-interest-ode) backwards in time for $V$, starting from its value $V(T)$ at time $T$. We will therefore be willing to pay no more than $V(0)$ for this guaranteed payment of $V(T)$ in the future. This value $V(0)$ is known as the ***present value***. The procedure of valuing, at an earlier time $t$, a payment received at time $T$ is known as ***discounting***.

    The solution at time $t$ to [(1.1.1)](#eq-interest-ode) with $V(T) = 1$ and interest rate $r$ is denoted $Z(t,T,r)$. This is the _discount factor_.

    We typically assume that interest rates are constant and compounded continuously; then $Z(t,T,r) = e^{-r(T-t)}$.

    In a similar spirit, we can define a ***compounding factor*** $B(t,T,r)$, which is the amount that a borrower of $\$ 1$ at time $t$ owes the lender at time $T$ when the interest rate is $r$. To calculate $B(t,T,r)$, we solve [(1.1.1)](#eq-interest-ode) forwards in time with $B(t,t,r) = 1$.
    When compounding continuously with a constant interest rate, $B(t,T,r) = e^{r(T-t)}$.

    By differentiating $ZB$, we can show that
    <span id="eq-discount-vs-compound"></span>
    $$Z(t,T,r)B(t,T,r) = 1.\tag{1.1.4}$$

    #### 1.1.2. Risk in Finance:
    To understand options, it is important to conceptually understand risk.

    Risk is a central concept in finance. The success of an organisation is often determined by how much return they can generate for a certain amount of risk they take on.

    We take the approach in [[Joshi 2003](#ref-MJ03)] and begin by looking at a _riskless_ asset. A riskless asset is an asset which has a precisely determined future value.

    An (approximate) example of a riskless asset would be a government bond.


    #### 1.1.3. Bonds:

    ***Definition:*** A cashflow can be thought of as a series of cash payments and expenses at set times.

    A ***bond*** is a contract in which the investor pays a price to acquire the right to receive regular payments of $c$, called the ***coupon***, from the issuer until $T$ years have elapsed. When this time has elapsed, the bond is said to have ***matured***. Upon maturity, the issuer pays the final coupon together with the ***principal*** (or ***face value***) $F$, so the final payment is $c + F$ [[Joshi 2003](#ref-MJ03), [Hull 2014, S4.4](#ref-JCH14)].

    We can also express the coupon as a percentage of the principal [[Wilmott 2006](#ref-PW06)].

    A ***zero-coupon bond*** is a bond with $c = 0$. Such bonds are useful for the theoretical and mathematical arguments used to derive results throughout this subject area, particularly when wanting to understand how the value of an asset changes over time. The present value of a zero-coupon bond at time $t$ with principal $F$, maturity date $T$ and interest rate $r$ is $FZ(t,T,r)$.

    A bond can be thought of as a portfolio of zero-coupon bonds.
    If a bond pays a coupon $c$ at times $t_{1} < t_{2} < \cdots < t_{n} < t_{n + 1} = T$ and has principal $F$, we can view the bond as a portfolio of $n+1$ zero-coupon bonds: $n$ with principal $c$ maturing at times $t_{1}, \ldots, t_{n}$, and one with principal $c+F$ maturing at time $T$.

    The value of the bond at time $t$ is then [[Hull 2014, S4.4](#ref-JCH14)]

    <span id="eq-bond-present-value"></span>
    $$V(t) = \sum_{i\leq n,\ t_{i}\geq t}cZ(t,t_{i},r) + (c+F)Z(t,T,r). \tag{1.1.5}$$

    ***Definition:*** The ***bond yield*** is the single rate which, when applied to all cashflows of the bond, gives a present value equal to its _market price_ [[Hull 2014, S4.4](#ref-JCH14)].

    A government bond is one issued by the government. In reality, there is risk attached to a government bond, due to uncertainty about interest rates. When interest rates decrease, bond holders benefit. Conversely, when interest rates increase, bond holders lose out.

    Government bonds can be traded on the market. The price of a bond reflects today's interest rates and the expectation of future interest rates, with bond prices moving in the opposite direction to the direction in which interest rates move.
    """)
    return


@app.class_definition
class InteractiveBond(mo.ui.dictionary):
    def __init__(
        self,
        max_principal: int = 1000,
        max_coupon: int = 100,
        max_maturity: int = 10,
        max_period: int = 2,
        max_price: int = 800,
    ):
        super().__init__(
            {
                "principal": mo.ui.slider(
                    start=0, stop=max_principal, value=max_principal // 4, step=1
                ),
                "coupon": mo.ui.slider(start=0, stop=max_coupon, value=0, step=1),
                "maturity": mo.ui.slider(
                    start=0, stop=max_maturity, value=max_maturity // 2, step=0.25
                ),
                "period": mo.ui.slider(
                    start=0.25, stop=max_period, value=max_period / 4, step=0.25
                ),  # > 0: cash_flows divides by it
                "interest_rate": mo.ui.slider(start=0, stop=1, value=0.05, step=0.01),
                "market_price": mo.ui.slider(
                    start=0, stop=max_price, value=max_price / 4, step=1
                ),
            }
        )

    @property
    def principal(self):
        return self["principal"]

    @property
    def coupon(self):
        return self["coupon"]

    @property
    def maturity(self):
        return self["maturity"]

    @property
    def period(self):
        return self["period"]

    @property
    def interest_rate(self):
        return self["interest_rate"]

    @property
    def market_price(self):
        return self["market_price"]

    def present_value_at(self, rate):
        # Present value of the bond's cashflows discounted at a given (continuously compounded) rate
        times, coupons, principal = self.cash_flows()
        return float(np.sum(np.exp(-rate * times) * (coupons + principal)))

    @property
    def ytm(self) -> float | None:
        # Yield to maturity: the single continuously compounded rate y with present_value_at(y) == market price
        # (the bond yield defined in Section 1.1.3). Returns a float, or None if no such rate exists.
        _, coupons, principal = self.cash_flows()
        total_cashflow = (coupons + principal).sum()
        if (
            self.market_price.value <= 0
            or self.maturity.value == 0
            or total_cashflow == 0
        ):
            return None

        price_err = lambda y: self.present_value_at(y) - self.market_price.value
        bracket: list[float] = [-1.0, 1.0]

        # Choosing bracket. Present value is decreasing, so we require price_err(bracket[0]) * price_err(bracket[1]) <= 0

        pv_a, pv_b = price_err(bracket[0]), price_err(bracket[1])

        while pv_a * pv_b > 0:
            if pv_a < 0:
                bracket[0] -= 1
            if pv_b > 0:
                bracket[1] += 1
            pv_a, pv_b = price_err(bracket[0]), price_err(bracket[1])

        ytm_soln = root_scalar(price_err, bracket=bracket, method="brentq")

        return ytm_soln.root if ytm_soln.converged else None

    def cash_flows(self):
        # Payment times and amounts received by the holder. Coupon dates are counted back from
        # maturity every `period` years (T, T - period, T - 2 period, ..., all > 0), so if T is not a
        # whole number of periods, the first coupon period is the short one. The principal is paid with the last coupon.
        T, p = self.maturity.value, self.period.value
        n = (
            int(np.floor(T / p - 1e-9)) + 1 if T > 0 else 0
        )  # number of coupon dates in (0, T]
        if n == 0:
            times = np.array([T])
            coupons = np.zeros(1)
        else:
            times = T - p * np.arange(n - 1, -1, -1)
            coupons = np.full(n, float(self.coupon.value))
        principal = np.zeros_like(coupons)
        principal[-1] = self.principal.value
        return times, coupons, principal

    def present_value(self):
        # Theoretical price: the bond's cashflows discounted at the (continuously compounded) interest rate
        return self.present_value_at(self.interest_rate.value)

    def display(self):
        html_rep = mo.vstack(
            [
                mo.md(r"##### Bond:"),
                mo.md(rf"Principal (Face Value): $\$ {self.principal.value}$"),
                self.principal,
                mo.md(rf"Coupon: $\$ {self.coupon.value}$"),
                self.coupon,
                mo.md(rf"Time Between Coupons: ${self.period.value}$ years"),
                self.period,
                mo.md(rf"Time Until Maturity: ${self.maturity.value}$ years"),
                self.maturity,
                mo.md(
                    rf"Interest Rate: ${self.interest_rate.value}$ ({self.interest_rate.value * 100:.0f}% per annum)"
                ),
                self.interest_rate,
                mo.md(rf"Market Price: $\$ {self.market_price.value}$"),
                self.market_price,
            ]
        )
        return html_rep

    def __str__(self):
        str_rep = "\n".join(
            [
                "Bond:",
                f"  Principal (Face Value): ${self.principal.value}",
                f"  Coupon: ${self.coupon.value}",
                f"  Time Between Coupons: {self.period.value} years",
                f"  Time Until Maturity: {self.maturity.value} years",
                f"  Interest Rate: {self.interest_rate.value} ({self.interest_rate.value * 100:.0f}% per annum)",
            ]
        )
        return str_rep


@app.cell
def _():
    example_bond = InteractiveBond()
    return (example_bond,)


@app.cell
def _(example_bond):
    bond_times, bond_coupons, bond_principal = example_bond.cash_flows()
    bond_discounted = np.exp(-example_bond.interest_rate.value * bond_times) * (
        bond_coupons + bond_principal
    )
    bond_bar_width = 0.6 * min(example_bond.period.value, 1)

    bond_cf_fig, bond_cf_ax = plt.subplots(figsize=(6, 4))
    bond_cf_ax.set_title(r"Bond Cashflow Received by the Holder")
    bond_cf_ax.set_xlabel(r"Time in years")
    bond_cf_ax.set_ylabel(r"Payment in \$")
    bond_cf_ax.axhline(0, color=theme.FOREGROUND, lw=1)
    bond_cf_ax.bar(
        bond_times,
        bond_coupons,
        width=bond_bar_width,
        color=theme.BLUE,
        label=r"Coupon",
    )
    bond_cf_ax.bar(
        bond_times,
        bond_principal,
        width=bond_bar_width,
        bottom=bond_coupons,
        color=theme.YELLOW,
        label=r"Principal",
    )
    bond_paid = bond_discounted > 0
    bond_cf_ax.plot(
        bond_times[bond_paid],
        bond_discounted[bond_paid],
        "o",
        color=theme.PINK,
        label=r"Discounted to $t = 0$",
    )
    bond_cf_ax.set_xlim(-0.5, max(example_bond.maturity.value, 1) + 0.5)
    bond_cf_ax.legend()

    bond_ytm = example_bond.ytm
    bond_ytm_text = (
        "undefined for these inputs"
        if bond_ytm is None
        else rf"${100 * bond_ytm:.2f}\%$ per annum (continuously compounded)"
    )

    mo.vstack(
        [
            example_bond.display(),
            xref.figure(
                __file__,
                "bond-cash-flows",
                bond_cf_fig,
                r"Cashflows received by the bond holder, and their values discounted to $t = 0$.",
            ),
            mo.md(
                rf"Present value of the cashflow (sum of the pink points): $\$ {bond_discounted.sum():.2f}$"
            ),
            mo.md(
                rf"Bond yield implied by the market price of $\$ {example_bond.market_price.value}$: {bond_ytm_text}"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-derivatives"></span>
    ### 1.2. Derivatives:

    ***Definition:*** A ***derivative*** is a financial instrument whose value depends on (i.e. is derived from) the values of other, more basic, underlying variables. Often a derivative's value depends on the prices of traded assets [[Hull 2014](#ref-JCH14)].

    Options are the main derivative we will be discussing in this notebook, but it helps to understand some other derivatives.

    ***Definition:*** A ***forward*** contract is an agreement to buy or sell an asset at a certain future time, called the time of ***maturity*** of the contract, for a certain price, called the ***delivery price*** [[Hull 2014](#ref-JCH14)].

    ***Definition:*** A ***futures*** contract is essentially a forward contract, but with certain features standardised to allow for easier exchange on a market [[Hull 2014](#ref-JCH14)].

    ***Definition:*** A ***spot*** contract is an agreement to buy or sell an asset almost immediately [[Hull 2014](#ref-JCH14)].

    One of the parties to a forward/futures contract agrees to buy the underlying asset at a predetermined future date at a specified price. This is called taking a ***long*** position on a forward/futures contract.

    The counterparty agrees to sell the asset at that predetermined date at the agreed upon price. This party is said to have taken a ***short*** position.

    Forward contracts are often used, for example, in foreign exchange markets [[Hull 2014](#ref-JCH14)].

    Understanding the payoff from forward/futures contracts is straightforward. Suppose the time of maturity is $T$, the price of the underlying asset at maturity is $S(T)$ and the delivery price is $K$.

    The side of the contract taking the long position must pay $K$ for the underlying asset, which is worth $S(T)$ on the market, so their payoff is given by

    <span id="eq-fwd-long"></span>
    $$S(T)-K, \tag{1.2.1}$$

    a similar line of reasoning can also be used to conclude that the party taking the short position has a payoff of

    <span id="eq-fwd-short"></span>
    $$K-S(T). \tag{1.2.2}$$

    We can display the payoffs as functions of the price of the underlying asset at the contract's maturity.
    """)
    return


@app.cell
def _():
    p_asset = np.linspace(0, 200, 1000)
    return (p_asset,)


@app.cell
def _():
    fwd_delivery_price = mo.ui.slider(start=10, stop=100, step=1, value=50)
    return (fwd_delivery_price,)


@app.cell
def _(fwd_delivery_price, p_asset):
    fwd_long_payoff = p_asset - fwd_delivery_price.value
    fwd_short_payoff = fwd_delivery_price.value - p_asset

    fwd_fig, fwd_ax = plt.subplots(figsize=(6, 4))
    fwd_ax.set_title(
        r"Forward/Futures Contract: Payoff in \$ vs Asset Price at Maturity"
    )
    fwd_ax.set_xlabel(r"Asset Price at Maturity in $\$$")
    fwd_ax.set_ylabel(r"Payoff in $\$$")
    fwd_ax.set_ylim(-200, 200)
    fwd_ax.axhline(0, color=theme.FOREGROUND, lw=1)
    fwd_ax.axvline(
        fwd_delivery_price.value, color=theme.PINK, lw=1, label=r"Delivery Price"
    )
    fwd_ax.plot(p_asset, fwd_long_payoff, label=r"Payoff (Long)")
    fwd_ax.plot(p_asset, fwd_short_payoff, label=r"Payoff (Short)")
    fwd_ax.legend()

    mo.vstack(
        [
            mo.md(rf"Delivery Price = $\$ {fwd_delivery_price.value}$"),
            fwd_delivery_price,
            xref.figure(
                __file__,
                "fwd-payoff",
                fwd_fig,
                r"Payoffs at maturity of the long and short positions in a forward contract, [(1.2.1)](#eq-fwd-long) and [(1.2.2)](#eq-fwd-short).",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-what-are-options"></span>
    ### 1.3. What Are Options?

    ##### **Definitions:**
    An **_option_** is a contract that grants the holder the _right_, but not the obligation, to buy or sell the underlying asset for a pre-specified price at or by a pre-specified time [[Investopedia](#ref-INV_op), [De Weert 2011](#ref-DW11), [Kwok 2008](#ref-YKK08)].

    The pre-specified price is called the ***strike price*** of the option.

    The pre-specified time is called the ***maturity*** or ***expiration time*** of the option.

    Options are an important financial instrument for risk management: they allow the holder to guard against certain events, and therefore reduce exposure to certain types of risk.

    ***Example:*** a livestock farmer needs hay to feed their animals. The supply of hay in any given year is dependent on the weather that year: poor weather for example reduces the amount of hay that is produced, which can lead to an increase in the price of hay. To protect against a sudden increase in the farmer's costs, the farmer may agree to a contract that gives them the right to buy hay at an agreed upon price with their supplier.

    We can categorise options into two types, based on what side of the transaction they give the holder rights over:
    * ***Call*** options give the holder the right to ***buy*** the underlying asset at the strike price at or by the expiration time.
    * ***Put*** options give the holder the right to ***sell*** the underlying asset at the strike price at or by the expiration time.

    When the holder of the option uses the right the option grants them, we say that the option has been ***exercised***.

    We can also categorise options based on when they can be exercised:
    * ***European*** options can only be exercised at maturity.
    * ***American*** options can be exercised at any time up until (and including) maturity.

    The counterparty to the holder of the option is called the _option writer_. They are obliged to fulfil the terms of the option contract if the holder chooses to exercise the option.
    We say that the holder takes a _**long**_ position on the option and the writer takes a _**short**_ position [[Kwok 2008](#ref-YKK08)].

    The interaction between the holder and writer is zero-sum: when one side of the transaction benefits, the other loses by the same amount.

    As the holder of the option has the choice to exercise the option or not, they will do so if it favours them. In the case of a call option, this is when the market price for the underlying asset is above the strike price, and for a put option, this is when the market price for the underlying asset is below the strike price.

    The writer of an option is exposed to risk whenever they write an option contract. This is because there is a chance the price of the underlying asset may become unfavourable for the writer (i.e. beneficial to the holder). To address the risk the writer takes on, they charge a _premium_, which is the cost of the option.

    American options can be exercised at a range of possible times up until maturity, unlike European options which can only be exercised at maturity. European options are thus simpler. Our examples in this notebook will therefore primarily focus on them [[Hull 2014](#ref-JCH14)].

    In an options market for a given underlying asset, there tend to be four types of market participants. Below is a table of them with their respective position in the market:

    |      | Writers         | Holders       |
    |------| --------------- | ------------- |
    | Put  | Shorting Puts   | Longing Puts  |
    | Call | Shorting Calls  | Longing Calls |
    """)
    return


@app.cell
def InteractiveOption():
    # Defining an interactive option's object.
    # It subclasses mo.ui.dictionary so that the option is itself a UI element:

    class InteractiveOption(mo.ui.dictionary):
        def __init__(
            self,
            side: str,
            min_strike: float = 0,
            max_strike: float = 100,
            max_maturity: float = 100,
            max_premium: float = 100,
            interest_rate: float = 0.05,
            strike: float | None = None,
            maturity: float | None = None,
            premium: float | None = None,
        ):
            if side not in ("call", "put"):
                raise ValueError(f"side must be 'call' or 'put', not {side!r}")
            super().__init__(
                {
                    "premium": mo.ui.slider(
                        start=1,
                        stop=max_premium,
                        value=max_premium / 4 if premium is None else premium,
                        step=1,
                    ),
                    "strike_price": mo.ui.slider(
                        start=min_strike,
                        stop=max_strike,
                        value=(max_strike - min_strike) / 4 + min_strike
                        if strike is None
                        else strike,
                        step=1,
                    ),
                    "maturity": mo.ui.slider(
                        start=0,
                        stop=max_maturity,
                        value=max_maturity / 4 if maturity is None else maturity,
                        step=1,
                    ),
                    "interest_rate": mo.ui.slider(
                        start=0, stop=1.0, value=interest_rate, step=0.01
                    ),
                }
            )
            self.side = side

        @property
        def premium(self):
            return self["premium"]

        @property
        def strike_price(self):
            return self["strike_price"]

        @property
        def maturity(self):
            return self["maturity"]

        @property
        def interest_rate(self):
            return self["interest_rate"]

        def holder_payoff(self, asset_price):
            if self.side == "call":
                return np.maximum(0, asset_price - self.strike_price.value)
            if self.side == "put":
                return np.maximum(0, self.strike_price.value - asset_price)

        def holder_profit(self, asset_price):
            return (
                self.holder_payoff(asset_price)
                - np.exp(self.interest_rate.value * self.maturity.value)
                * self.premium.value
            )

        def writer_payoff(self, asset_price):
            return -self.holder_payoff(asset_price)

        def writer_profit(self, asset_price):
            return -self.holder_profit(asset_price)

        def display(self, name: str = ""):
            html_rep = mo.vstack(
                [
                    mo.md(
                        rf"##### {name + ' -- ' if name else name} {self.side.title()} Option:"
                    ),
                    mo.md(rf"Premium: $\$ {self.premium.value}$"),
                    self.premium,
                    mo.md(rf"Strike Price: $\$ {self.strike_price.value}$"),
                    self.strike_price,
                    mo.md(rf"Time Until Maturity: ${self.maturity.value}$ years"),
                    self.maturity,
                    mo.md(
                        rf"Interest Rate: ${self.interest_rate.value}$ ({self.interest_rate.value * 100:.0f}% per annum)"
                    ),
                    self.interest_rate,
                ]
            )
            return html_rep

        def __str__(self):
            str_rep = "\n".join(
                [
                    f"{self.side.title()} Option:",
                    f"  Premium: ${self.premium.value}",
                    f"  Strike Price: ${self.strike_price.value}",
                    f"  Time Until Maturity: {self.maturity.value} years",
                    f"  Interest Rate: {self.interest_rate.value} ({self.interest_rate.value * 100:.0f}% per annum)",
                ]
            )
            return str_rep

    class InteractiveEuropeanOption(InteractiveOption):
        # Exercised only at maturity, so the payoff and profit formulas above apply as they are
        pass

    return (InteractiveEuropeanOption,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-european-call"></span>
    ### 1.4. Example – The European Call Option:
    Suppose there is an asset and a European call option for this asset.

    Furthermore, suppose that the writer for this option:
    * sets the strike price at $K$
    * and sets the expiration to be $T$ years from now.

    The option is European, so it can only be exercised at that time. At expiration, the market price of the asset is $S(T)$.

    We will assume there are no transaction costs associated with the purchase of the asset or the option, and that the interest is compounded continuously at a (risk-free) rate of $r$ per annum.

    What is the option worth to the holder at expiry? This is how much is gained from having the option and it is called the ***payoff*** of the option.
    We can answer this question by looking at the three possible scenarios:
    1. ***Out of the money (OTM):*** The market price of the asset at expiration is less than the strike price, $S(T) < K$:
       * The holder chooses not to exercise their option. This is because they can purchase the underlying asset on the market for a price less than the strike price.
       * The option is therefore worthless; the payoff is $0$.
    2. ***At the money (ATM):*** The market price of the asset at expiration is equal to the strike price, $S(T) = K$:
       * It makes no difference whether the holder exercises the option or not as there is no difference between purchasing the underlying asset from the writer or the market.
       * The option is therefore worthless; the payoff is $0$.
    3. ***In the money (ITM):*** The market price of the asset at expiration is greater than the strike price, $S(T) > K$:
       * The holder chooses to exercise their option. This is because they can purchase the underlying asset from the writer at the strike price $K$, then sell the asset on the market for $S(T)$.
       * The payoff of the option in this case is $S(T)-K$.

    Putting this together, the payoff for the holder is therefore given by:

    <span id="eq-call-payoff-cases"></span>
    $$\text{Payoff}_{\text{holder}} = \begin{cases} 0&\text{ if }S(T)\le K\cr S(T) - K &\text{ if }S(T) > K\end{cases}, \tag{1.4.1}$$

    which we can write as

    <span id="eq-call-payoff"></span>
    $$\text{Payoff}_{\text{holder}} = \max(S(T) - K, 0). \tag{1.4.2}$$

    We can also consider the writer's perspective of the option. Because the option contract is zero-sum, whatever the holder gains, the writer loses. Therefore the payoff for the writer is given by

    <span id="eq-call-payoff-writer"></span>
    $$\text{Payoff}_{\text{writer}} = -\max(S(T) - K, 0). \tag{1.4.3}$$

    The writer's payoff is never positive, and they take on risk when writing the option. The writer therefore charges the holder a premium, say $C(0)$ (the price of the call at time $0$), which is the cost of the option contract itself.

    ***Note:*** when referring to the option's payoff, we are typically referring to the holder's payoff unless otherwise stated.

    To calculate the P&L at expiration for the holder and writer, we therefore need to account for the payoff and the option premium. The writer received $C(0)$ upon their sale of the option. Over the $T$ years, this compounds to $C(0)B(0,T,r)$. The option holder paid $C(0)$ for the option, but upon maturity, this loss becomes $C(0)B(0,T,r)$. The extra loss incurred can be thought of as the opportunity cost of not having $C(0)$ to compound over time.

    The holder's and writer's P&L upon maturity of the option are therefore:

    <span id="eq-call-pnl-holder"></span>
    $$\text{P\&L}_{\text{holder}} = \max(S(T) - K, 0) - C(0)B(0,T,r) \tag{1.4.4}$$

    <span id="eq-call-pnl-writer"></span>
    $$\text{P\&L}_{\text{writer}} = C(0)B(0,T,r) - \max(S(T) - K, 0). \tag{1.4.5}$$

    We plot the P&L functions for both the holder and the writer below:
    """)
    return


@app.cell
def _(InteractiveEuropeanOption):
    euro_call = InteractiveEuropeanOption(
        "call", max_maturity=10, premium=50, strike=50, maturity=1
    )
    return (euro_call,)


@app.cell
def _(euro_call):
    euro_call.display()
    return


@app.cell
def _(euro_call, p_asset):
    euro_call_strike = euro_call.strike_price.value
    ex1_pnl_holder = euro_call.holder_profit(p_asset)
    ex1_pnl_writer = euro_call.writer_profit(p_asset)

    ex1fig, ex1ax = plt.subplots(figsize=(6, 4))
    ex1ax.set_title(r"European Call Option: Net Profit in \$ vs Asset Price")
    ex1ax.set_xlabel(r"Asset Price in $\$$")
    ex1ax.set_ylabel(r"Net Profit in $\$$")
    ex1_ylim = 1.1 * max(np.abs(ex1_pnl_holder).max(), 1)
    ex1ax.set_ylim(-ex1_ylim, ex1_ylim)
    ex1ax.axhline(0, color=theme.FOREGROUND, lw=1)
    ex1ax.axvline(euro_call_strike, color=theme.PINK, lw=1, label=r"Strike Price")
    ex1ax.plot(p_asset, ex1_pnl_holder, label=r"P\&L (Holder)")
    ex1ax.plot(p_asset, ex1_pnl_writer, label=r"P\&L (Writer)")
    ex1ax.legend()
    xref.figure(
        __file__,
        "call-pnl",
        ex1fig,
        r"Net profit at maturity of the holder and writer of a European call, [(1.4.4)](#eq-call-pnl-holder) and [(1.4.5)](#eq-call-pnl-writer).",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    From the diagram above, we can see that for the holder:
    * Their P&L is bounded below: the most they can lose is $C(0)B(0,T,r)$.
    * Their potential net profit is unbounded because $S(T)$ is unbounded above.

    In contrast, the writer:
    * Can potentially incur an unbounded loss.
    * Has a maximum net profit of $C(0)B(0,T,r)$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-european-put"></span>
    ### 1.5. Example – The European Put Option:

    We can utilise similar logic for European put options, but instead of the holder purchasing the underlying asset, the holder is _selling_ it.

    Suppose again that we have an asset and a writer creates a European put option for this asset where they:
    * set the strike price at $K$
    * and set the expiration to be $T$ years from now.

    The option can only be exercised at expiration as it is a European option. At expiration, the market price of the asset is $S(T)$.

    As with the previous example, we will assume there are no transaction costs associated with the purchase of the asset or the option, and that the interest is compounded continuously at a (risk-free) rate of $r$ per annum.

    What is the payoff of the option? We answer this by looking at three different cases:
    1. ***Out of the money (OTM):*** The market price of the asset at expiration is greater than the strike price, $S(T) > K$:
       * The holder chooses not to exercise their option. This is because they can sell the underlying asset for more on the market.
       * The option is therefore worthless; the payoff is $0$.
    2. ***At the money (ATM):*** The market price of the asset at expiration is equal to the strike price, $S(T) = K$:
       * It makes no difference whether the holder exercises the option or not as there is no difference between selling to the writer or to the market.
       * The option is therefore worthless; the payoff is $0$.
    3. ***In the money (ITM):*** The market price of the asset at expiration is less than the strike price, $S(T) < K$:
       * The holder chooses to exercise the option. They can purchase the underlying asset from the market at $S(T)$ and sell it to the writer at the strike price $K$.
       * The payoff of the option is thus $K-S(T)$.

    Putting this together, the payoff for the holder of a European put option is given by:

    <span id="eq-put-payoff-cases"></span>
    $$\text{Payoff}_{\text{holder}} = \begin{cases} 0&\text{ if }S(T)\ge K\cr K - S(T) &\text{ if }S(T) < K\end{cases}, \tag{1.5.1}$$

    which we can write as

    <span id="eq-put-payoff"></span>
    $$\text{Payoff}_{\text{holder}} = \max(K - S(T), 0). \tag{1.5.2}$$

    Similarly to the European call option example, whatever the holder gains, the writer loses. Therefore the payoff for the writer is given by

    <span id="eq-put-payoff-writer"></span>
    $$\text{Payoff}_{\text{writer}} = -\max(K - S(T), 0). \tag{1.5.3}$$

    The writer therefore charges the holder a premium, $P(0)$ (the price of the put at time $0$), to compensate for the risk involved in writing the option.

    Accounting for interest, the holder's and writer's P&L at the time of maturity of the European put option are:

    <span id="eq-put-pnl-holder"></span>
    $$\text{P\&L}_{\text{holder}} = \max(K - S(T), 0) - P(0)B(0,T,r) \tag{1.5.4}$$

    <span id="eq-put-pnl-writer"></span>
    $$\text{P\&L}_{\text{writer}} = P(0)B(0,T,r) - \max(K - S(T), 0). \tag{1.5.5}$$

    We plot the P&L functions for both the holder and the writer below:
    """)
    return


@app.cell
def _(InteractiveEuropeanOption):
    euro_put = InteractiveEuropeanOption(
        "put", max_maturity=10, premium=50, strike=50, maturity=1
    )
    return (euro_put,)


@app.cell
def _(euro_put):
    euro_put.display()
    return


@app.cell
def _(euro_put, p_asset):
    euro_put_strike = euro_put.strike_price.value

    ex2_pnl_holder = euro_put.holder_profit(p_asset)
    ex2_pnl_writer = euro_put.writer_profit(p_asset)

    ex2fig, ex2ax = plt.subplots(figsize=(6, 4))
    ex2ax.set_title(r"European Put Option: Net Profit in \$ vs Asset Price")
    ex2ax.set_xlabel(r"Asset Price in $\$$")
    ex2ax.set_ylabel(r"Net Profit in $\$$")
    ex2_ylim = 1.1 * max(np.abs(ex2_pnl_holder).max(), 1)
    ex2ax.set_ylim(-ex2_ylim, ex2_ylim)
    ex2ax.axhline(0, color=theme.FOREGROUND, lw=1)
    ex2ax.axvline(euro_put_strike, color=theme.PINK, lw=1, label=r"Strike Price")
    ex2ax.plot(p_asset, ex2_pnl_holder, label=r"P\&L (Holder)")
    ex2ax.plot(p_asset, ex2_pnl_writer, label=r"P\&L (Writer)")
    ex2ax.legend()
    xref.figure(
        __file__,
        "put-pnl",
        ex2fig,
        r"Net profit at maturity of the holder and writer of a European put, [(1.5.4)](#eq-put-pnl-holder) and [(1.5.5)](#eq-put-pnl-writer).",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Unlike the European call option, the P&L for both sides of the option contract is bounded.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-moneyness"></span>
    ### 1.6. Intrinsic Value and Moneyness:

    ***Definition:*** The ***intrinsic value*** of an option is the _payoff_ were the option to be exercised immediately [[De Weert 2011](#ref-DW11)].

    ***Definition:*** The ***moneyness*** of an option is a measure of an option's intrinsic value obtained by comparing the option's strike price and the underlying asset's market price [[Investopedia](#ref-INV_moneyness)].

    We can quantify the moneyness by taking the ratio $\frac{S}{K}$. (**Note:** other conventions for quantifying moneyness exist.)

    In the examples above, to understand the payoff of an option we looked at the possible states the option could be in: OTM, ATM and ITM.

    | **Option** | $100x\%$ **ITM**                     | **ATM**                          | $100x\%$ **OTM**                     |
    | ---------- | ------------------------------------ | -------------------------------- | ------------------------------------ |
    |    Call    | $\frac{S}{K} = 1 + x$ | $\frac{S}{K} = 1$ | $\frac{S}{K} = 1 - x$ |
    |    Put     | $\frac{S}{K} = 1 - x$ | $\frac{S}{K} = 1$ | $\frac{S}{K} = 1 + x$ |

    Here, $x > 0$ quantifies to what extent the option is ITM/OTM (with $x \le 1$ in the put ITM and call OTM cases, as $S \ge 0$).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-use-of-options"></span>
    ### 1.7. The Use of Options:

    There are three broad categories of traders in options markets (and derivative markets more broadly) [[Hull 2014](#ref-JCH14)]:
    * Hedgers
    * Speculators
    * Arbitrageurs

    These reflect the three main uses of options (and derivatives more broadly): hedging, speculation and arbitrage.

    #### 1.7.1. Speculating:
    ***Speculators*** use derivatives to bet on the future direction of a market variable, seeking to make a return, potentially taking on more risk in doing so [[Hull 2014](#ref-JCH14)].

    ***Example:*** Suppose that a speculator believes that an asset's price is likely to increase over the next $T$ years. The asset price is currently $S(0)$ and a $T$-year _call_ option with a strike price of $K$ is selling for $C(0)$.

    Suppose the speculator wishes to invest a total of $W$. They can allocate some amount of their capital to _call_ options or the underlying asset.

    Below we graph three speculation strategies: one where the speculator invests all of their capital into the underlying asset, another where the speculator invests all their capital into call options, and a third where the speculator adopts a mixed strategy: allocating some of their capital to call options, and the rest to the underlying asset.
    """)
    return


@app.cell
def _(InteractiveEuropeanOption):
    spec_call = InteractiveEuropeanOption(
        "call", max_maturity=10, premium=50, strike=50, maturity=1
    )

    spec_asset = mo.ui.dictionary(
        {"Initial Price": mo.ui.slider(start=1, stop=100, value=25, step=1)}
    )

    total_investment = mo.ui.slider(start=100, stop=100000, value=25000, step=100)
    return spec_asset, spec_call, total_investment


@app.cell
def _(total_investment):
    # Start at about a quarter of the total, rounded to the slider's step of 100 (total_investment is itself a multiple of 100)
    call_investment = mo.ui.slider(
        start=0,
        stop=total_investment.value,
        value=100 * round(total_investment.value / 400),
        step=100,
    )
    return (call_investment,)


@app.cell
def _(call_investment, spec_asset, spec_call, total_investment):
    mo.vstack(
        [
            mo.md("#### Speculation Example Properties"),
            mo.md(
                rf"Asset Initial Price: $\$ {spec_asset['Initial Price'].value:.2f}$"
            ),
            spec_asset["Initial Price"],
            mo.md("#### Strategy Properties"),
            mo.hstack(
                [
                    mo.md(rf"Total Investment: $\$ {total_investment.value:.2f}$"),
                    total_investment,
                    mo.md(
                        rf"Investment in Call Options: $\$ {call_investment.value:.2f}$"
                    ),
                    call_investment,
                    mo.md(
                        rf"Investment in Asset: $\$ {total_investment.value - call_investment.value:.2f}$"
                    ),
                ]
            ),
            spec_call.display(),
        ]
    )
    return


@app.cell
def _(call_investment, spec_asset, spec_call, total_investment):
    # Plot up to twice the larger of the strike and the initial price, so the call's payoff region is always visible
    spec_S_T = np.linspace(
        0, 2 * max(spec_call.strike_price.value, spec_asset["Initial Price"].value), 500
    )
    call_profit_per_dollar = spec_call.holder_profit(spec_S_T) / spec_call.premium.value
    asset_profit_per_dollar = spec_S_T / spec_asset["Initial Price"].value - np.exp(
        spec_call.interest_rate.value * spec_call.maturity.value
    )

    call_wgt = call_investment.value / total_investment.value
    asset_wgt = 1.0 - call_wgt

    strat = call_wgt * call_profit_per_dollar + asset_wgt * asset_profit_per_dollar

    strat_profit = total_investment.value * strat
    fully_call_profit = total_investment.value * call_profit_per_dollar
    fully_asset_profit = total_investment.value * asset_profit_per_dollar

    spec_fig, spec_ax = plt.subplots(figsize=(6, 4))
    spec_ax.plot(
        spec_S_T,
        strat_profit,
        label=rf"Strategy {100 * call_wgt:.2f}\% Call Options, {100 * asset_wgt:.2f}\% Asset",
    )
    spec_ax.plot(spec_S_T, fully_call_profit, label=r"100\% Call Options")
    spec_ax.plot(spec_S_T, fully_asset_profit, label=r"100\% Asset")
    spec_ax.axhline(y=0, color=theme.FOREGROUND, linewidth=1)

    spec_ax.set_ylabel(r"Net Profit in \$")
    spec_ax.set_xlabel(r"Underlying Asset Price at Maturity in \$")
    spec_ax.set_title(r"Speculation Strategies: Net Profit in \$")
    spec_ax.legend()

    xref.figure(
        __file__,
        "speculation",
        spec_fig,
        r"Net profit at maturity of the speculation strategy, compared with investing everything in call options or in the asset.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As seen in the graph above, when the speculator adds more options to their portfolio, they can potentially be much more profitable than only purchasing the underlying asset. They can however incur a greater loss. Speculation with options leads to the investor taking on more risk. This is known as the _leveraging effect_ [[Pascucci 2011](#ref-AP11)].
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.7.2. Hedging:

    ***Hedgers*** use derivatives to _reduce the risk_ that they face from potential future price movements in the market [[Hull 2014](#ref-JCH14)].

    ***Example:*** Suppose an investor has purchased a total of $N$ units of an asset at a price $S(0)$ per unit for their portfolio, but they are concerned about the possibility of the asset's price declining over a period of $T$ years. For simplicity, let us assume that the interest rate is a constant $r$ over the period of time.

    The investor could buy _put_ options to protect against this possibility.

    Suppose the _put_ option gives the investor the right to sell the asset at a price of $K$ per unit, and the premium is $P(0)$ per unit. This hedging strategy costs $NP(0)B(0,T,r)$ (valued at maturity), but guarantees that the units can be sold for at least $K$ each at maturity, so the amount realised is at least $N(K - P(0)B(0,T,r))$. If the market price $S(T)$ stays above $K$ then the options are not exercised and thus are worthless, but in this case the total value of the units is $NS(T)$. Accounting for the cost of the options, the net value of the portfolio is $N(S(T)-P(0)B(0,T,r))$.

    Below we show the net value of the portfolio as a function of $S(T)$, the price of the asset at maturity.
    """)
    return


@app.cell
def _(InteractiveEuropeanOption):
    hedge_put = InteractiveEuropeanOption(
        "put", max_maturity=10, max_premium=20, premium=10, strike=20, maturity=2
    )
    hedge_num_shares = mo.ui.slider(start=1, stop=1000, value=100, step=1)
    return hedge_num_shares, hedge_put


@app.cell
def _(hedge_num_shares, hedge_put):
    mo.vstack(
        [
            mo.md("#### Hedging Strategy"),
            mo.md(rf"Number of Units: $N = {hedge_num_shares.value}$"),
            hedge_num_shares,
            hedge_put.display(),
        ]
    )
    return


@app.cell
def _(hedge_num_shares, hedge_put):
    hedge_S_T = np.linspace(0, 100, 500)
    holding = hedge_num_shares.value * hedge_S_T
    hedge_strat = hedge_num_shares.value * (
        hedge_S_T + hedge_put.holder_profit(hedge_S_T)
    )

    hedge_fig, hedge_ax = plt.subplots(figsize=(6, 4))
    hedge_ax.plot(hedge_S_T, holding, label="No Hedging")
    hedge_ax.plot(hedge_S_T, hedge_strat, label="Hedging")
    hedge_ax.axhline(0, color=theme.FOREGROUND, linewidth=1)
    hedge_ax.set_xlim(0, np.max(hedge_S_T))

    hedge_ax.set_title(
        rf"Value of the Example Portfolio of N = {hedge_num_shares.value} Units in \$"
    )
    hedge_ax.set_ylabel(r"Portfolio Value in \$")
    hedge_ax.set_xlabel(r"Underlying Asset Price at Maturity in \$")
    hedge_ax.legend()

    xref.figure(
        __file__,
        "hedging",
        hedge_fig,
        r"Value at maturity of the example portfolio, with and without the protective puts.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.7.3. Arbitrage:

    ***Arbitrageurs*** take offsetting positions in two or more instruments to lock in a riskless profit [[Hull 2014](#ref-JCH14)]. The arbitrageur achieves their profit by spotting mispricings of financial instruments [[Joshi 2003](#ref-MJ03)].

    Broadly speaking, arbitrage is the possibility of carrying out a financial operation without any investment, but leading to a profit without any risk [[Pascucci 2011](#ref-AP11)].

    ***Example:*** Suppose there is an asset that is traded in two different markets, Market A and Market B. Suppose the price in Market A is $S^{A}$ and the price in Market B is $S^{B}$ with $S^{A} < S^{B}$. An arbitrageur simultaneously buys the asset in Market A and sells it in Market B and makes a profit, assuming negligible transaction costs [[Hull 2014](#ref-JCH14)].

    Arbitrage opportunities do not last very long. Once found, many market participants flock to the opportunity: buying from Market A and selling in Market B. The increased demand in Market A drives up the price of the asset in this market and the increased supply in Market B drives the price there down. For this reason, it tends to be assumed that no arbitrage opportunities exist in large enough financial markets [[Hull 2014](#ref-JCH14)].

    If two financial instruments will __certainly__ have the same value at a future date, then, in an arbitrage-free market, they must have the same price at the current moment. If their prices differed, an arbitrage opportunity would arise by selling the more expensive instrument and buying the cheaper one [[Pascucci 2011](#ref-AP11)].

    Using a similar argument, we can express the no-arbitrage principle in the following way: where $X(t)$ and $Y(t)$ are the values at time $t$ of two financial instruments,

    <span id="eq-no-arbitrage"></span>
    $$X(T)\leq Y(T)\implies X(t)\leq Y(t),\qquad t\leq T. \tag{1.7.1}$$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-assumptions"></span>
    ### 1.8. Common Assumptions in Mathematical Finance [[Joshi 2003](#ref-MJ03)]:
    1. Not moving the market:
       * Our actions do not affect the market price.
       * Therefore we can buy or sell any amount without affecting the price.
       * This assumption is not true in general, but is a reasonable approximation if we are trading quantities which are small relative to the overall size of the market.
    2. Liquidity:
       * We can buy or sell as much as we wish at the market price at any given time.
       * This assumption does not hold in all markets, but is a reasonable assumption to make in large markets with many participants, as it is more likely someone is willing to buy or sell.
    3. Shorting:
       * Shorting can happen at will.
       * Mathematically, shorting is having negative amounts of an asset.
       * In reality, this means selling an asset that one does not hold.
       * Some markets may put restrictions on short-selling assets.
    4. Non-whole number quantities:
       * Of course, it's usually not feasible to own a fraction of an asset, but when dealing with extremely large quantities, the non-integer part of a quantity is small enough that this error doesn't matter.
       * This usually makes the maths easier (we can assume continuity).
    5. Negligible transaction costs:
       * This means one can buy and sell assets without the costs associated with making the transaction having an impact.
       * There are two ways one can incur transaction costs:
            - The act of buying and selling costs money (paying the market a fee to use their services)
            - Buy and sell prices differ slightly; this is the _bid-offer spread_.
       * ***Remark:*** The size of the bid-offer spread is related to liquidity: in a highly liquid market it is tiny, whereas in a less liquid market it can be substantial. Bid-offer spreads are how banks make money [[Joshi 2003](#ref-MJ03)].
    6. The risk-free interest rate $r > 0$, and borrowing and lending at this interest rate is possible.
    7. No arbitrage opportunities are available.

    Depending on the model, other assumptions may be made, but typically, these seven assumptions are made, and when models fail, these assumptions should be questioned.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-strategies"></span>
    ### 1.9. Strategies Involving Options:

    We can allocate our capital to take a variety of positions in the market for different instruments and assets. How we do so is called a ***strategy***.

    A ***bullish*** strategy is one where the investor benefits if the underlying _increases_ in value.
    A ***bearish*** strategy is one where the investor benefits if the underlying _decreases_ in value [[De Weert 2011, Ch.10](#ref-DW11)], [[Hull 2014](#ref-JCH14)].

    Mathematically, the strategy's payoff (or net profit) is given by summing the payoffs (or net profits) of the individual instruments and assets employed in the strategy.

    #### 1.9.1. Options and the Underlying Asset:

    We have four types of strategies [[Hull 2014](#ref-JCH14)]:
    * A ***covered call***: this involves going _long_ in the underlying asset and taking a _short_ position in a corresponding European _call_ option.
    * The _reverse_ of a _covered call_, which involves taking a _short_ position in the underlying asset and a _long_ position in a corresponding European _call_ option.
    * A ***protective put***: this involves going long on the underlying asset and going _long_ on a European _put_ option.
    * The _reverse_ of a _protective put_, which involves going _short_ in both the underlying asset and a European _put_ option.
    """)
    return


@app.function
def payoff_diagram(asset_price, legs, total, title, strategy_only=False):
    # Payoff diagram in the style of the call spread below: each leg's payoff dashed, its strike(s) as
    # dotted vertical lines in the same colour, and the strategy's total payoff as a solid yellow line.
    # legs: (label, payoff, strikes) triples, where strikes is a number, a list of numbers, or None.
    # strategy_only: hide the legs' payoffs and draw each distinct strike once, in one neutral colour.
    leg_colours = theme.CHALK[1:]  # yellow is kept for the strategy's payoff
    if len(legs) > len(leg_colours):
        raise ValueError(
            f"at most {len(leg_colours)} legs can be drawn, not {len(legs)}"
        )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(y=0, color=theme.FOREGROUND, lw=1)
    leg_strikes = [
        [] if strikes is None else np.atleast_1d(strikes) for _, _, strikes in legs
    ]
    if strategy_only:
        for i, strike in enumerate(
            sorted({float(k) for ks in leg_strikes for k in ks})
        ):
            ax.axvline(
                x=strike,
                color=theme.MUTED,
                lw=1,
                linestyle=":",
                label="Strikes" if i == 0 else None,
            )
    else:
        for (label, payoff, _), strikes, colour in zip(legs, leg_strikes, leg_colours):
            ax.plot(
                asset_price, payoff, color=colour, linestyle="--", label=label, zorder=3
            )
            for strike in strikes:
                ax.axvline(x=strike, color=colour, lw=1, linestyle=":")
    # Drawn beneath the legs, so a leg that coincides with the total stays visible
    ax.plot(
        asset_price, total, color=theme.YELLOW, lw=2, label="Strategy Payoff", zorder=2
    )
    ax.set_ylabel(r"Payoff in \$")
    ax.set_xlabel(r"Underlying Asset Price at Expiry in \$, $S(T)$")
    ax.set_title(title)
    ax.legend()
    return fig


@app.cell
def _(InteractiveEuropeanOption):
    stock_strategy = mo.ui.radio(
        options=[
            "Covered Call",
            "Reverse Covered Call",
            "Protective Put",
            "Reverse Protective Put",
        ],
        value="Covered Call",
    )
    stock_call = InteractiveEuropeanOption(
        "call", max_strike=150, strike=100, max_maturity=10, maturity=1
    )
    stock_put = InteractiveEuropeanOption(
        "put", max_strike=150, strike=100, max_maturity=10, maturity=1
    )
    stock_payoff_only = mo.ui.switch(label="Strategy payoff only", value=False)
    return stock_call, stock_payoff_only, stock_put, stock_strategy


@app.cell
def _(stock_call, stock_put, stock_strategy):
    # (sign of the underlying asset position, option, sign of the option position), where +1 is long and -1 is short
    stock_legs = {
        "Covered Call": (1, stock_call, -1),
        "Reverse Covered Call": (-1, stock_call, 1),
        "Protective Put": (1, stock_put, 1),
        "Reverse Protective Put": (-1, stock_put, -1),
    }[stock_strategy.value]
    mo.hstack(
        [
            mo.vstack([mo.md("##### Strategy:"), stock_strategy]),
            stock_legs[1].display("Option"),
        ]
    )
    return (stock_legs,)


@app.cell
def _(p_asset, stock_legs, stock_payoff_only, stock_strategy):
    _stock_sign, _option, _option_sign = stock_legs
    _position = {1: "Long", -1: "Short"}
    _stock_payoff = _stock_sign * p_asset
    _option_payoff = _option_sign * _option.holder_payoff(p_asset)
    _fig = payoff_diagram(
        p_asset,
        legs=[
            (f"{_position[_stock_sign]} Underlying Asset Payoff", _stock_payoff, None),
            (
                f"{_position[_option_sign]} {_option.side.title()} Payoff",
                _option_payoff,
                _option.strike_price.value,
            ),
        ],
        total=_stock_payoff + _option_payoff,
        title=f"{stock_strategy.value} Payoff",
        strategy_only=stock_payoff_only.value,
    )
    xref.figure(
        __file__,
        "stock-and-option",
        mo.vstack([stock_payoff_only, _fig]),
        r"Payoff at expiry of the selected position in the underlying asset and one option.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.9.2. Call Spread:

    A ***call spread*** involves nothing more than two calls with different strike prices but the same maturity.
    An investor is said to be ***buying the call spread*** if they buy the call with the lower strike price and sell the call with the higher strike price; this is a _bullish_ strategy. ***Selling the call spread*** means selling the call with the lower strike price and buying the call with the higher strike price; this is a _bearish_ strategy.
    The payoff for this strategy is bounded [[De Weert 2011](#ref-DW11)].
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.9.3. Put Spread:

    Buying a ***put spread*** means buying a higher strike put and selling a lower strike put.
    A put spread can be thought of as the opposite of a call spread. It is bearish when bought, and bullish when sold.

    Since a put with a higher strike is never cheaper than a put with a lower strike and the same maturity (see [Table 2.1.1](#tab-price-properties) in Section 2.1), the investor pays a premium for this strategy. The payoff for this strategy is bounded [[De Weert 2011](#ref-DW11)].
    """)
    return


@app.cell
def _(InteractiveEuropeanOption):
    # One diagram shared by Sections 1.9.2 and 1.9.3. Each spread keeps its own options,
    # so a reader's slider settings for one spread survive switching to the other.
    spread_type = mo.ui.radio(
        options=["Call Spread", "Put Spread"], value="Call Spread"
    )
    # Call spread: hold the lower strike call (Call 1), write the higher strike call (Call 2)
    call_1_bull_spread = InteractiveEuropeanOption(
        "call", min_strike=0, max_strike=75, strike=40
    )
    call_2_bull_spread = InteractiveEuropeanOption(
        "call", min_strike=75, max_strike=150, strike=110
    )
    # Put spread: hold the higher strike put (Put 1), write the lower strike put (Put 2)
    put_1_bear_spread = InteractiveEuropeanOption(
        "put", min_strike=75, max_strike=150, strike=110
    )
    put_2_bear_spread = InteractiveEuropeanOption(
        "put", min_strike=0, max_strike=75, strike=40
    )
    spread_payoff_only = mo.ui.switch(label="Strategy payoff only", value=False)
    return (
        call_1_bull_spread,
        call_2_bull_spread,
        put_1_bear_spread,
        put_2_bear_spread,
        spread_payoff_only,
        spread_type,
    )


@app.cell
def _(
    call_1_bull_spread,
    call_2_bull_spread,
    put_1_bear_spread,
    put_2_bear_spread,
    spread_type,
):
    # (held option, written option, name used in the labels) for the selected spread
    spread_legs = {
        "Call Spread": (call_1_bull_spread, call_2_bull_spread, "Call"),
        "Put Spread": (put_1_bear_spread, put_2_bear_spread, "Put"),
    }[spread_type.value]
    mo.vstack(
        [
            mo.vstack([mo.md("##### Spread (Sections 1.9.2-1.9.3):"), spread_type]),
            mo.hstack(
                [
                    spread_legs[0].display(f"{spread_legs[2]} 1 (Held)"),
                    spread_legs[1].display(f"{spread_legs[2]} 2 (Written)"),
                ]
            ),
        ]
    )
    return (spread_legs,)


@app.cell
def _(p_asset, spread_legs, spread_payoff_only, spread_type):
    _held, _written, _name = spread_legs
    _held_payoff = _held.holder_payoff(p_asset)
    _written_payoff = _written.writer_payoff(p_asset)
    _fig = payoff_diagram(
        p_asset,
        legs=[
            (f"Holding {_name} 1 Payoff", _held_payoff, _held.strike_price.value),
            (f"Writing {_name} 2 Payoff", _written_payoff, _written.strike_price.value),
        ],
        total=_held_payoff + _written_payoff,
        title=f"{spread_type.value} Payoff",
        strategy_only=spread_payoff_only.value,
    )
    xref.figure(
        __file__,
        "spreads",
        mo.vstack([spread_payoff_only, _fig]),
        r"Payoff at expiry of the selected spread.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.9.4. Box Spread:

    A ***box spread*** is a combination of a bull call spread with strike prices $K_{1}<K_{2}$ and a bear put spread with the same two strike prices.

    The payoff from such a strategy is always $K_{2} - K_{1}$ at expiry, so the strategy is worth $(K_{2} - K_{1})Z(0,T,r)$ today. If the four options can be bought or sold for a different total price, there is an arbitrage opportunity.

    Note: Box spread arbitrage only works with European options [[Hull 2014](#ref-JCH14)].
    """)
    return


@app.cell
def _():
    box_lower_strike = mo.ui.slider(start=0, stop=75, value=40, step=1)
    box_higher_strike = mo.ui.slider(start=75, stop=150, value=110, step=1)
    box_payoff_only = mo.ui.switch(label="Strategy payoff only", value=False)
    return box_higher_strike, box_lower_strike, box_payoff_only


@app.cell
def _(box_higher_strike, box_lower_strike, box_payoff_only, p_asset):
    _K1, _K2 = box_lower_strike.value, box_higher_strike.value
    _bull_call_spread = np.maximum(p_asset - _K1, 0) - np.maximum(p_asset - _K2, 0)
    _bear_put_spread = np.maximum(_K2 - p_asset, 0) - np.maximum(_K1 - p_asset, 0)
    mo.vstack(
        [
            mo.md(rf"Lower Strike: $K_1 = \$ {_K1}$"),
            box_lower_strike,
            mo.md(rf"Higher Strike: $K_2 = \$ {_K2}$"),
            box_higher_strike,
            xref.figure(
                __file__,
                "box-spread",
                mo.vstack(
                    [
                        box_payoff_only,
                        payoff_diagram(
                            p_asset,
                            legs=[
                                (
                                    r"Bull Call Spread Payoff (long $K_1$ call, short $K_2$ call)",
                                    _bull_call_spread,
                                    [_K1, _K2],
                                ),
                                (
                                    r"Bear Put Spread Payoff (long $K_2$ put, short $K_1$ put)",
                                    _bear_put_spread,
                                    [_K1, _K2],
                                ),
                            ],
                            total=_bull_call_spread + _bear_put_spread,
                            title=rf"Box Spread Payoff: $K_2 - K_1 = {_K2 - _K1}$",
                            strategy_only=box_payoff_only.value,
                        ),
                    ]
                ),
                r"Payoff at expiry of the box spread: a bull call spread plus a bear put spread (the switch hides these legs).",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.9.5. Butterfly Spread:

    A ***butterfly spread*** involves positions in options with three different strike prices. It can be created by buying a European call option with a relatively low strike price $K_{1}$, a European call option with a relatively high strike price $K_{3}$, and selling two European call options with a strike price $K_{2}$ which lies between $K_{1}$ and $K_{3}$, usually halfway, or near the current price of the underlying asset.
    A butterfly spread leads to a greater payoff if the price of the underlying asset stays close to $K_{2}$, but a smaller payoff if there is a significant move in its price in either direction. It's therefore a strategy used by investors who believe that large movements in the price of the underlying asset are unlikely.

    We can also follow the reverse strategy to get the corresponding short strategy. This strategy produces a moderate net profit if there is a large movement in price [[Hull 2014](#ref-JCH14)].
    """)
    return


@app.cell
def _(InteractiveEuropeanOption):
    fly_position = mo.ui.radio(
        options=["Long Butterfly", "Short Butterfly"], value="Long Butterfly"
    )
    fly_call_1 = InteractiveEuropeanOption(
        "call", min_strike=0, max_strike=66, strike=40
    )
    fly_call_2 = InteractiveEuropeanOption(
        "call", min_strike=67, max_strike=133, strike=90
    )  # halfway between K1 and K3
    fly_call_3 = InteractiveEuropeanOption(
        "call", min_strike=134, max_strike=200, strike=140
    )
    fly_payoff_only = mo.ui.switch(label="Strategy payoff only", value=False)
    return fly_call_1, fly_call_2, fly_call_3, fly_payoff_only, fly_position


@app.cell
def _(fly_call_1, fly_call_2, fly_call_3, fly_position):
    mo.vstack(
        [
            mo.vstack([mo.md("##### Position:"), fly_position]),
            mo.hstack(
                [
                    fly_call_1.display("Call 1"),
                    fly_call_2.display("Call 2"),
                    fly_call_3.display("Call 3"),
                ]
            ),
        ]
    )
    return


@app.cell
def _(
    fly_call_1,
    fly_call_2,
    fly_call_3,
    fly_payoff_only,
    fly_position,
    p_asset,
):
    # The long butterfly holds calls 1 and 3 and writes two of call 2; the short butterfly is the reverse
    _sign = 1 if fly_position.value == "Long Butterfly" else -1
    _outer, _inner = ("Holding", "Writing") if _sign == 1 else ("Writing", "Holding")
    _leg_1 = _sign * fly_call_1.holder_payoff(p_asset)
    _leg_2 = -2 * _sign * fly_call_2.holder_payoff(p_asset)
    _leg_3 = _sign * fly_call_3.holder_payoff(p_asset)
    _fig = payoff_diagram(
        p_asset,
        legs=[
            (f"{_outer} Call 1 Payoff", _leg_1, fly_call_1.strike_price.value),
            (
                rf"{_inner} $2\times$ Call 2 Payoff",
                _leg_2,
                fly_call_2.strike_price.value,
            ),
            (f"{_outer} Call 3 Payoff", _leg_3, fly_call_3.strike_price.value),
        ],
        total=_leg_1 + _leg_2 + _leg_3,
        title=f"{fly_position.value} Spread Payoff",
        strategy_only=fly_payoff_only.value,
    )
    xref.figure(
        __file__,
        "butterfly",
        mo.vstack([fly_payoff_only, _fig]),
        r"Payoff at expiry of the selected butterfly spread.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.9.6. Collar:

    A ***collar*** strategy consists of a call with a higher strike and a put with a lower strike, with opposite positions held in both.
    If an investor is very bullish on the underlying asset and thinks its downside is limited, they could buy the call and sell the put to fund this strategy.
    The diagram shows that the payoff is unbounded: in principle, there is no upper limit to the investor's payoff, but they simultaneously expose themselves to the price of the underlying asset going down.
    The investor is said to be ***gearing*** their view because they can lose more than their initial investment if the price of the underlying asset ends up lower than the strike price of the put option.

    If, instead, the investor is feeling bearish on the underlying asset, they could buy the put and sell the call to fund the strategy, but they expose themselves to the price of the underlying asset going up [[De Weert 2011](#ref-DW11)].
    """)
    return


@app.cell
def _(InteractiveEuropeanOption):
    collar_view = mo.ui.radio(
        options=[
            "Bullish (buy the call, sell the put)",
            "Bearish (buy the put, sell the call)",
        ],
        value="Bullish (buy the call, sell the put)",
    )
    collar_call = InteractiveEuropeanOption(
        "call", min_strike=75, max_strike=150, strike=110
    )
    collar_put = InteractiveEuropeanOption(
        "put", min_strike=0, max_strike=75, strike=40
    )
    collar_payoff_only = mo.ui.switch(label="Strategy payoff only", value=False)
    return collar_call, collar_payoff_only, collar_put, collar_view


@app.cell
def _(collar_call, collar_put, collar_view):
    mo.vstack(
        [
            mo.vstack([mo.md("##### View:"), collar_view]),
            mo.hstack(
                [
                    collar_call.display("Higher Strike"),
                    collar_put.display("Lower Strike"),
                ]
            ),
        ]
    )
    return


@app.cell
def _(collar_call, collar_payoff_only, collar_put, collar_view, p_asset):
    _bullish = collar_view.value.startswith("Bullish")
    _call_leg = (
        collar_call.holder_payoff(p_asset)
        if _bullish
        else collar_call.writer_payoff(p_asset)
    )
    _put_leg = (
        collar_put.writer_payoff(p_asset)
        if _bullish
        else collar_put.holder_payoff(p_asset)
    )
    _fig = payoff_diagram(
        p_asset,
        legs=[
            (
                f"{'Holding' if _bullish else 'Writing'} Call Payoff",
                _call_leg,
                collar_call.strike_price.value,
            ),
            (
                f"{'Writing' if _bullish else 'Holding'} Put Payoff",
                _put_leg,
                collar_put.strike_price.value,
            ),
        ],
        total=_call_leg + _put_leg,
        title=f"{'Bullish' if _bullish else 'Bearish'} Collar Payoff",
        strategy_only=collar_payoff_only.value,
    )
    xref.figure(
        __file__,
        "collar",
        mo.vstack([collar_payoff_only, _fig]),
        r"Payoff at expiry of the selected collar.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.9.7. Straddle:

    The ***straddle*** consists of buying a call option and a put option with the same strike and maturity. It is a popular strategy for investors who are confident the underlying asset will move from its current price, but are uncertain about the direction of that movement [[De Weert 2011](#ref-DW11)].
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.9.8. Strangle:

    The ***strangle*** is similar to the straddle. It consists of buying a call option and a put option, but the call and put need not have the same strike price. It consists of a call with a higher strike and a put with a lower strike.

    It is also useful to investors who are confident that the underlying asset will move away from current levels, but wish to take on less risk. The strangle is less risky than the straddle, but the potential net profit is less as well [[De Weert 2011](#ref-DW11)].
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.9.9. Strips and Straps:

    Strips and straps are similar to straddles. A ***strip*** consists of a long position in one European call and two European puts with the same strike price and expiration date. A ***strap*** consists of a long position in two European calls and a European put with the same strike price and expiration date.

    These strategies are used by investors who strongly believe there will be a big move in the price of the underlying asset, and have some confidence in the direction of the price movement [[Hull 2014](#ref-JCH14)].
    """)
    return


@app.cell
def _():
    # One diagram shared by Sections 1.9.7-1.9.9. The strike sliders live here so their values persist when the strategy changes.
    vol_strategy = mo.ui.radio(
        options=["Straddle", "Strangle", "Strip", "Strap"], value="Straddle"
    )
    vol_strike = mo.ui.slider(start=0, stop=200, value=100, step=1)
    vol_put_strike = mo.ui.slider(start=0, stop=100, value=70, step=1)
    vol_call_strike = mo.ui.slider(start=100, stop=200, value=130, step=1)
    vol_payoff_only = mo.ui.switch(label="Strategy payoff only", value=False)
    return (
        vol_call_strike,
        vol_payoff_only,
        vol_put_strike,
        vol_strategy,
        vol_strike,
    )


@app.cell
def _(vol_strategy):
    # Recreated whenever the strategy changes, so the counts reset to that strategy's definition: (calls, puts)
    _counts = {
        "Straddle": (1, 1),
        "Strangle": (1, 1),
        "Strip": (1, 2),
        "Strap": (2, 1),
    }[vol_strategy.value]
    vol_num_calls = mo.ui.slider(start=1, stop=5, value=_counts[0], step=1)
    vol_num_puts = mo.ui.slider(start=1, stop=5, value=_counts[1], step=1)
    return vol_num_calls, vol_num_puts


@app.cell
def _(
    vol_call_strike,
    vol_num_calls,
    vol_num_puts,
    vol_put_strike,
    vol_strategy,
    vol_strike,
):
    _controls = [mo.md("##### Strategy (Sections 1.9.7-1.9.9):"), vol_strategy]
    if vol_strategy.value == "Strangle":
        _controls += [
            mo.md(rf"Put Strike (lower): $\$ {vol_put_strike.value}$"),
            vol_put_strike,
            mo.md(rf"Call Strike (higher): $\$ {vol_call_strike.value}$"),
            vol_call_strike,
        ]
    else:
        _controls += [
            mo.md(
                rf"Strike Price (shared by the calls and puts): $\$ {vol_strike.value}$"
            ),
            vol_strike,
        ]
    if vol_strategy.value in ("Strip", "Strap"):
        _controls += [
            mo.md(rf"Number of Calls Bought: ${vol_num_calls.value}$"),
            vol_num_calls,
            mo.md(rf"Number of Puts Bought: ${vol_num_puts.value}$"),
            vol_num_puts,
        ]
    mo.vstack(_controls)
    return


@app.cell
def _(
    p_asset,
    vol_call_strike,
    vol_num_calls,
    vol_num_puts,
    vol_payoff_only,
    vol_put_strike,
    vol_strategy,
    vol_strike,
):
    _strangle = vol_strategy.value == "Strangle"
    _call_K = vol_call_strike.value if _strangle else vol_strike.value
    _put_K = vol_put_strike.value if _strangle else vol_strike.value
    # Straddles and strangles hold exactly one call and one put; only strips and straps use the count sliders
    _n_calls, _n_puts = (
        (vol_num_calls.value, vol_num_puts.value)
        if vol_strategy.value in ("Strip", "Strap")
        else (1, 1)
    )
    _calls = _n_calls * np.maximum(p_asset - _call_K, 0)
    _puts = _n_puts * np.maximum(_put_K - p_asset, 0)
    _plural = lambda n, word: f"{n} {word}" + ("s" if n != 1 else "")
    _fig = payoff_diagram(
        p_asset,
        legs=[
            (f"Holding {_plural(_n_calls, 'Call')} Payoff", _calls, _call_K),
            (f"Holding {_plural(_n_puts, 'Put')} Payoff", _puts, _put_K),
        ],
        total=_calls + _puts,
        title=f"{vol_strategy.value} Payoff: Long {_plural(_n_calls, 'Call')} and {_plural(_n_puts, 'Put')}",
        strategy_only=vol_payoff_only.value,
    )
    xref.figure(
        __file__,
        "volatility-strategies",
        mo.vstack([vol_payoff_only, _fig]),
        r"Payoff at expiry of the selected straddle, strangle, strip or strap.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1.9.10. Strategies Involving Options with Differing Expiration Dates:

    We can also employ strategies involving options with different expiry dates. Some examples of such strategies are:
    * Calendar spreads: where the options used in the strategy have the same strike price but different expiry dates. For example, we can create a calendar spread by selling a European call option with a strike price $K$ and buying a longer-maturity European call option with the same strike price. On expiry of the shorter-maturity option, we close the position of the longer-maturity option.
    * Diagonal spreads: Both the expiration date and the strike price of the calls involved in the strategy are different.

    In theory, we can combine all sorts of positions in an option market, holding positions in options of different types, strikes, and expirations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-forward-price"></span>
    ### 1.10. The Forward Price:

    The delivery price of a forward is chosen so that the value of the forward contract to both parties is zero at the time when the contract is initiated. This delivery price is called the ***forward price*** [[Kwok 2008](#ref-YKK08)].

    Suppose $f(t,T,S,K)$ denotes the value at time $t$ of a long position in a forward contract with maturity $T$ and delivery price $K$ on an underlying asset with price $S$, and $K_{*}(t,T,S)$ denotes the forward price of the underlying asset at time $t$.

    If we assume the underlying asset pays no dividends, consider the following two portfolios:
    * Portfolio A: a long position in a forward contract on the underlying asset with maturity $T$ and delivery price $K$, and a zero-coupon bond with principal $K$ that matures at time $T$.
    * Portfolio B: one unit of the underlying asset.

    The value of Portfolio A at time $t$ is $f(t,T,S,K) + KZ(t,T,r)$, and the value of Portfolio B is $S(t)$.
    At time $T$, the value of Portfolio A is $f(T,T,S,K) + K$, and the value of Portfolio B is $S(T)$. From [(1.2.1)](#eq-fwd-long), $f(T,T,S,K) = S(T) - K$, so at maturity Portfolios A and B both have the value $S(T)$. Using [(1.7.1)](#eq-no-arbitrage), we have
    $$f(t,T,S,K) + KZ(t,T,r) = S(t),$$
    i.e.
    $$f(t,T,S,K) = S(t) - KZ(t,T,r).$$
    The forward price is the value of the delivery price $K$ that makes $f = 0$. Therefore, the forward price is given by
    <span id="eq-fwd-price"></span>
    $$K_{*}(t,T,S) = S(t)B(t,T,r), \tag{1.10.1}$$
    where we have used $B(t,T,r)Z(t,T,r) = 1$ [(1.1.4)](#eq-discount-vs-compound).

    If we assume dividends are paid discretely, let $D(t)$ be the present value at time $t$ of all dividends paid by the asset over the remaining time until the maturity of the forward contract. We modify the argument used in the non-dividend-paying case by having Portfolio B consist of one unit of the asset plus a borrowing of $D(t)$.

    <span id="rmk-dividend-effect"></span>
    ***Remark:*** In cases where dividend payments are discrete and the instrument's payoff can only occur at maturity, the effect of dividend payments is essentially the substitution $S \to S - D$ to get from the non-dividend-paying case to the dividend-paying case.

    As a result, the forward price in the dividend-paying case is
    <span id="eq-fwd-price-div"></span>
    $$K_{*}(t,T,S) = (S(t) - D(t))B(t,T,r). \tag{1.10.2}$$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-price-properties"></span>
    ## 2. Properties of the Prices of Options:

    In this section, we explore the mathematical properties of the functions that give the price of American and European options. We let:
    * $t$ be the current time
    * $T$ be the time of expiry
    * $K$ be the strike price
    * $S$ be the price of the underlying asset at time $t$ (written $S(t)$ when the time needs to be explicit)
    * $C_{E}(t, T, S, K)$ be the price at time $t$ of a European call option with expiry at time $T$, underlying asset price $S$ and strike price $K$.
    * $C_{A}(t, T, S, K)$ be the price at time $t$ of an American call option with expiry at time $T$, underlying asset price $S$ and strike price $K$.
    * $P_{E}(t, T, S, K)$ be the price at time $t$ of a European put option with expiry at time $T$, underlying asset price $S$ and strike price $K$.
    * $P_{A}(t, T, S, K)$ be the price at time $t$ of an American put option with expiry at time $T$, underlying asset price $S$ and strike price $K$.

    When the dependence on $t$, $T$, $S$, $K$ and $r$ is unambiguous, we omit these arguments, writing for example $C_{E}$ and $Z$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-price-bounds"></span>
    ### 2.1. Bounds on the Price of Options:

    For this section, we will use the assumptions in [Section 1.8](#sec-assumptions) as well as assuming that the underlying asset pays no dividends.

    In particular, the assumption of no-arbitrage is important in deriving many of these results.

    Proofs for these results mostly utilise no-arbitrage arguments and clever portfolio construction. More details on how these results are proven can be found in [[Kwok 2008](#ref-YKK08), [Joshi 2003](#ref-MJ03), and [Hull 2014](#ref-JCH14)]:
    """)
    return


@app.cell(hide_code=True)
def _():
    xref.table(
        __file__,
        "price-properties",
        mo.md(r"""
    | Variable Increasing | European Call Price $C_{E}(t,T,S,K)$   | European Put Price $P_{E}(t,T,S,K)$    | American Call Price $C_{A}(t,T,S,K)$   | American Put Price $P_{A}(t,T,S,K)$    |
    | ------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
    |         $T$         | N/A                                    | N/A                                    | Increasing function                    | Increasing function                    |
    |         $S$         | Increasing function                    | Decreasing function                    | Increasing function                    | Decreasing function                    |
    |         $K$         | Decreasing, Convex, Lipschitz function | Increasing, Convex, Lipschitz function | Decreasing, Convex, Lipschitz function | Increasing, Convex, Lipschitz function |
    """),
        r"How the price of each option changes as one variable increases, with the other variables held fixed.",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ***Key Inequalities:*** These inequalities assume that no dividends are paid on the underlying asset over the lifetime of the option.

    <span id="eq-bounds-us-vs-eu"></span>
    $$\begin{aligned}C_{E}&\leq C_{A}\\ P_{E}&\leq P_{A}\end{aligned} \tag{2.1.1}$$

    <span id="eq-bounds-eu-call"></span>
    $$\max(S - KZ(t,T,r), 0)\leq C_{E}(t,T,S,K)\leq S \tag{2.1.2}$$

    <span id="eq-bounds-eu-put"></span>
    $$\max(KZ(t,T,r) - S, 0)\leq P_{E}(t,T,S,K)\leq KZ(t,T,r) \tag{2.1.3}$$

    <span id="eq-bounds-us-call"></span>
    $$\max(S-KZ(t,T,r),0)\leq C_{A}(t,T,S,K)\leq S \tag{2.1.4}$$

    <span id="eq-bounds-us-put"></span>
    $$\max(K-S,0)\leq P_{A}(t,T,S,K)\leq K \tag{2.1.5}$$

    ***Remark:*** We often assume _time-homogeneity_. This assumption means that the time $t$ does not have a direct effect on the price of the options: the rules of finance stay the same throughout time.
    In this case, we can express the price of an option in terms of the time until expiry, $T-t$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-put-call-parity"></span>
    ### 2.2. Put-Call Parity:

    The put-call parity describes the relationship between the prices of European put and call options with the same strike price, time until maturity and underlying asset price.

    Consider two portfolios:
    * Portfolio A: composed of one European call option with expiration date $T$ and strike price $K$ and one zero-coupon bond with principal $K$ that matures at time $T$.
    * Portfolio B: composed of one European put option with expiration date $T$ and strike price $K$, and the option's underlying asset whose price is $S$.

    The value of Portfolio A at a time $t\leq T$ is $C_{E}(t,T,S,K) + KZ(t,T,r)$, while Portfolio B is worth $P_{E}(t,T,S,K) + S$.

    At time $T$, Portfolios A and B are both worth $\max(S(T), K)$. By [(1.7.1)](#eq-no-arbitrage), both portfolios have equal value. Therefore
    <span id="eq-put-call-parity"></span>
    $$C_{E} + KZ = P_{E} + S.\tag{2.2.1}$$

    ***Remark:*** Put-call parity only holds for European options. For American options, the equivalent relationship when there are no dividends is:
    <span id="eq-us-put-call-relation"></span>
    $$S - K\leq C_{A} - P_{A}\leq S - KZ,\tag{2.2.2}$$
    as shown in [[Hull 2014](#ref-JCH14)].
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-dividends"></span>
    ### 2.3. The Effect of Dividends:

    If the underlying asset is a stock, there is the possibility that between the current time and expiration date, the stock pays a dividend.

    The dividend is a cash payment to the holder of a stock [[Investopedia](#ref-INV_dividend)].

    We assume that the dividends that are paid during the lifetime of the option are known. We will let $D$ denote the present value of all the dividends during the lifetime of the option. The proofs of the effects of dividends can be found in [[Joshi 2003](#ref-MJ03), [Hull 2014](#ref-JCH14)].

    The lower bounds in [(2.1.2)](#eq-bounds-eu-call) and [(2.1.3)](#eq-bounds-eu-put) for European call and put options change to:
    $$\begin{aligned} C_{E}&\geq\max(S - D - KZ, 0)\\ P_{E}&\geq\max(D + KZ - S, 0)\end{aligned}$$

    And the put-call relations in [(2.2.1)](#eq-put-call-parity) and [(2.2.2)](#eq-us-put-call-relation) become:
    $$C_{E} + D + KZ = P_{E} + S$$
    $$S - D - K\leq C_{A} - P_{A} \leq S - KZ$$
    """)
    return


@app.cell
def _():
    # Added by Claude: controls for the put-call parity figure below (Sections 2.2-2.3).
    # The rate starts at 0.01 because Section 1.8 assumes r > 0.
    parity_strike = mo.ui.slider(start=10, stop=150, value=100, step=1)
    parity_time = mo.ui.slider(start=0, stop=10, value=2, step=0.25)
    parity_rate = mo.ui.slider(start=0.01, stop=0.2, value=0.05, step=0.01)
    parity_dividend = mo.ui.slider(start=0, stop=30, value=0, step=1)
    return parity_dividend, parity_rate, parity_strike, parity_time


@app.cell
def _(p_asset, parity_dividend, parity_rate, parity_strike, parity_time):
    # Added by Claude: put-call parity (2.2.1) and the American relation (2.2.2), with their
    # dividend-adjusted forms from Section 2.3. Only C - P is plotted: parity fixes it without
    # a pricing model, whereas C and P on their own need one (Part III).
    _K, _tau, _r, _D = (
        parity_strike.value,
        parity_time.value,
        parity_rate.value,
        parity_dividend.value,
    )
    _Z = np.exp(-_r * _tau)
    _european = p_asset - _D - _K * _Z
    _american_low = p_asset - _D - _K
    _american_high = p_asset - _K * _Z

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.axhline(0, color=theme.FOREGROUND, lw=1)
    _ax.axvline(_K, color=theme.PINK, lw=1, linestyle=":", label=r"Strike Price $K$")
    _ax.fill_between(
        p_asset,
        _american_low,
        _american_high,
        color=theme.GREEN,
        alpha=0.3,
        lw=0,
        label=r"American: $S - D - K \le C_{A} - P_{A} \le S - KZ$",
    )
    _ax.plot(
        p_asset,
        _european,
        color=theme.YELLOW,
        lw=2,
        label=r"European: $C_{E} - P_{E} = S - D - KZ$",
    )
    _ax.plot(
        p_asset,
        p_asset - _K,
        color=theme.MUTED,
        lw=1,
        linestyle="--",
        label=r"Payoff difference at expiry: $S(T) - K$",
    )
    _ax.set_xlabel(r"Underlying Asset Price in \$, $S$")
    _ax.set_ylabel(r"Call Price minus Put Price in \$")
    _ax.set_title(r"Put-Call Parity: $C - P$ vs Asset Price")
    _ax.legend(loc="upper left")

    mo.vstack(
        [
            mo.md(rf"Strike Price: $K = \$ {_K}$"),
            parity_strike,
            mo.md(rf"Time Until Expiry: $T - t = {_tau}$ years"),
            parity_time,
            mo.md(rf"Interest Rate: $r = {_r:.2f}$ ({_r * 100:.0f}% per annum)"),
            parity_rate,
            mo.md(rf"Present Value of Dividends: $D = \$ {_D}$"),
            parity_dividend,
            mo.md(
                rf"Discount factor $Z = e^{{-r(T-t)}} = {_Z:.4f}$; width of the American band $K(1-Z) + D = {_K * (1 - _Z) + _D:.2f}$."
            ),
            xref.figure(
                __file__,
                "parity-band",
                _fig,
                r"The difference between the prices of a call and a put with the same strike and expiry, against the asset price $S$. For European options it is the line given by put-call parity [(2.2.1)](#eq-put-call-parity); for American options it lies in the band given by [(2.2.2)](#eq-us-put-call-relation). Dividends ($D > 0$) shift both as in [Section 2.3](#sec-dividends). The band has width $K(1-Z) + D$; with $D = 0$ the European line is its upper edge, and the band closes as $T - t \to 0$.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-early-exercise"></span>
    ### 2.4. Early Exercise of American Options [[Hull 2014](#ref-JCH14)]:

    It is never optimal to exercise an American call option on a non-dividend-paying stock before the expiration date.

    From inequalities [(2.1.1)](#eq-bounds-us-vs-eu) to [(2.1.5)](#eq-bounds-us-put), and the fact that the assumptions $r>0$ and $0\leq t< T$ imply $Z(t,T,r) < 1$, we can show that
    <span id="eq-early-exercise-call"></span>
    $$C_{A}\geq C_{E}\geq S-KZ> S-K. \tag{2.4.1}$$
    This means that the price of an American call option is always greater than the option's intrinsic value prior to maturity (assuming no arbitrage).

    If it were optimal to exercise an American call option prior to maturity, $C_{A}$ would equal the option's intrinsic value at that time. It follows that it is never optimal to exercise an American call option early. As it is never optimal to exercise an American call option early when the underlying stock is non-dividend-paying, the extra rights of the American call option, compared with its European equivalent, give it no extra value; therefore $C_{E}=C_{A}$. This can, however, be shown more formally [[Mittal 2026](#ref-SM26)].

    The two reasons why an American call option on a non-dividend-paying stock should not be exercised early are:
    * The time value of money: from the perspective of the option holder, the later the strike price is paid, the better.
    * A call option, when held instead of the underlying stock itself, insures the holder against the stock price falling below the strike price. Once the option has been exercised, the strike price has been exchanged for the stock price, and this insurance disappears.

    For American put options on a non-dividend-paying stock, it can sometimes be optimal to exercise early.
    This can be seen using inequalities [(2.1.1)](#eq-bounds-us-vs-eu) to [(2.1.5)](#eq-bounds-us-put) and assuming $r>0$. We have
    $$\max(K-S,0)\leq P_{A}\leq K.$$

    A put option, held with the underlying stock, insures the holder against the stock price falling below the strike price. It may be optimal for an investor to forgo this insurance and exercise the option early to realise the strike price immediately. As $S$ decreases, $r$ increases or the volatility of the underlying asset decreases, the early exercise of a put option becomes more attractive. In particular, when $S=0$, exercising the put option is optimal, as it gives the largest possible payoff; waiting can do no better.

    When dividend payments are expected, however, it may be optimal to exercise an American call option early. It may be optimal prior to an ex-dividend date, and not optimal at other times.

    ***Note:*** a stock going ex-dividend means that the company's dividend allocations have been specified, so buyers after a specified date do not receive the declared dividends. The ex-dividend date is the date on which the stock begins trading without the value of the next dividend [[Investopedia](#ref-INV_exdividend)].
    """)
    return


@app.cell
def _():
    # Added by Claude: controls for the early-exercise figure below (Section 2.4).
    # The rate starts at 0.01 because Section 1.8 assumes r > 0.
    exercise_strike = mo.ui.slider(start=10, stop=150, value=100, step=1)
    exercise_time = mo.ui.slider(start=0, stop=10, value=2, step=0.25)
    exercise_rate = mo.ui.slider(start=0.01, stop=0.3, value=0.1, step=0.01)
    exercise_dividend = mo.ui.slider(start=0, stop=50, value=0, step=1)
    return exercise_dividend, exercise_rate, exercise_strike, exercise_time


@app.cell
def _(exercise_dividend, exercise_rate, exercise_strike, exercise_time, p_asset):
    # Added by Claude: the no-arbitrage regions (2.1.1)-(2.1.5) for call and put prices,
    # compared with intrinsic value. Only proven bounds are drawn, not prices: the price
    # curves and the optimal exercise boundary need a pricing model (Part III).
    _K, _tau, _r, _D = (
        exercise_strike.value,
        exercise_time.value,
        exercise_rate.value,
        exercise_dividend.value,
    )
    _Z = np.exp(-_r * _tau)
    _gap = _K * (
        1 - _Z
    )  # K(1 - Z): the call's margin over intrinsic value, and the put threshold
    _call_intrinsic = np.maximum(p_asset - _K, 0)
    _put_intrinsic = np.maximum(_K - p_asset, 0)
    # max(S - D - KZ, 0) <= C_E <= C_A <= S, with the dividend-adjusted lower bound from Section 2.3
    _call_low = np.maximum(p_asset - _D - _K * _Z, 0)

    _fig, (_call_ax, _put_ax) = plt.subplots(1, 2, figsize=(12, 5))
    for _ax in (_call_ax, _put_ax):
        _ax.axhline(0, color=theme.FOREGROUND, lw=1)
        _ax.axvline(
            _K, color=theme.PINK, lw=1, linestyle=":", label=r"Strike Price $K$"
        )
        _ax.set_xlabel(r"Underlying Asset Price in \$, $S$")
        _ax.set_ylabel(r"Option Price in \$")

    _call_ax.fill_between(
        p_asset,
        _call_low,
        p_asset,
        color=theme.BLUE,
        alpha=0.3,
        lw=0,
        label=r"$\max(S - D - KZ, 0) \le C_{E} \le C_{A} \le S$",
    )
    _call_ax.plot(
        p_asset,
        _call_intrinsic,
        color=theme.YELLOW,
        lw=2,
        label=r"Intrinsic value $\max(S - K, 0)$",
    )
    # S - D - KZ < S - K exactly when D > K(1 - Z): only then can the lower bound fall below intrinsic value
    if _D > _gap:
        _call_ax.fill_between(
            p_asset,
            _call_low,
            _call_intrinsic,
            where=_call_intrinsic > _call_low,
            color=theme.PINK,
            alpha=0.5,
            lw=0,
            label=r"Lower bound below intrinsic value: early exercise not ruled out",
        )
    _call_ax.set_title(r"Call Options")
    _call_ax.legend(loc="upper left")

    _put_ax.fill_between(
        p_asset,
        _put_intrinsic,
        _K,
        color=theme.GREEN,
        alpha=0.25,
        lw=0,
        label=r"$\max(K - S, 0) \le P_{A} \le K$",
    )
    _put_ax.fill_between(
        p_asset,
        np.maximum(_K * _Z - p_asset, 0),
        _K * _Z,
        color=theme.BLUE,
        alpha=0.3,
        lw=0,
        label=r"$\max(KZ - S, 0) \le P_{E} \le KZ$",
    )
    _put_ax.plot(
        p_asset,
        _put_intrinsic,
        color=theme.YELLOW,
        lw=2,
        label=r"Intrinsic value $\max(K - S, 0)$",
    )
    # For S < K(1 - Z): P_E <= KZ < K - S <= P_A, by (2.1.3) and (2.1.5)
    if _gap > 0:
        _put_ax.axvspan(
            0,
            _gap,
            color=theme.PINK,
            alpha=0.25,
            lw=0,
            label=r"$S < K(1-Z)$: $P_{E} \le KZ < K - S \le P_{A}$",
        )
    _put_ax.set_ylim(-0.05 * _K, 1.8 * _K)  # headroom above K for the legend
    _put_ax.set_title(r"Put Options")
    _put_ax.legend(loc="upper right")
    _fig.tight_layout()

    _call_status = (
        rf"$D = {_D} > K(1-Z)$, so the call's lower bound falls below intrinsic value for $S > K$ (pink)."
        if _D > _gap
        else rf"$D = {_D} \le K(1-Z)$, so the call's lower bound stays at or above intrinsic value."
    )
    mo.vstack(
        [
            mo.md(rf"Strike Price: $K = \$ {_K}$"),
            exercise_strike,
            mo.md(rf"Time Until Expiry: $T - t = {_tau}$ years"),
            exercise_time,
            mo.md(rf"Interest Rate: $r = {_r:.2f}$ ({_r * 100:.0f}% per annum)"),
            exercise_rate,
            mo.md(rf"Present Value of Dividends (call only): $D = \$ {_D}$"),
            exercise_dividend,
            mo.md(
                rf"Discount factor $Z = e^{{-r(T-t)}} = {_Z:.4f}$ and $K(1-Z) = {_gap:.2f}$. "
                + _call_status
            ),
            xref.figure(
                __file__,
                "early-exercise-bounds",
                _fig,
                r"No-arbitrage regions for option prices against the asset price $S$, from [(2.1.1)](#eq-bounds-us-vs-eu) to [(2.1.5)](#eq-bounds-us-put), compared with intrinsic value. Left: with no dividends, for $S > K$ the call region lies at least $K(1-Z)$ above intrinsic value, so exercising early gives up value [(2.4.1)](#eq-early-exercise-call); with dividends $D > K(1-Z)$ (lower bound from [Section 2.3](#sec-dividends)) the lower bound falls below intrinsic value, so early exercise is no longer ruled out (pink). Right: for $S < K(1-Z)$ (pink), the European put must be worth less than immediate exercise, so the right to exercise early has strictly positive value. The regions are bounds, not prices.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    bib.bibliography(__file__)
    return


if __name__ == "__main__":
    app.run()
