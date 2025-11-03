# 2thamoon - a crypto sandbox simulator

[![Streamlit](https://img.shields.io/badge/built_with-Streamlit-ff4b4b.svg)](https://streamlit.io/)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

**Try it:** **https://2thamoon.streamlit.app/**

Simulate the mess. 2thamoon lets you crank the knobs on hype, liquidity, whales, attackers, and treasury policy to watch markets rip, stall, or implode. with 300+ things to mess with i reckon its impossible to see/do it all

---



hi guys. so im studying biz so i decided fuck it i wanna learn about crypto and how markets work. I'll be honest i didnt know shit about them so i decided to hand the project off to codex with my specs. I have read over the code a few times but its very vibe coded i want to preface with that. however its not vibe tested, ive used it and iterated on it a ton. it works decently imo. its built on streamlit so its actually really bug free the only issues and places where the sim has weird edge cases. There are so so so many things to mess with. from modifying growth, to algos, to making a stable coin, to defending against pump and dumps, to causing pump and dumbs, so so much. this is not a game however, its a literal simulator. imo its really fun. 

## quick start
clone repo then:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

(if you have streamlit skip venv)

## features 
i will however note this started off as a way for me to sim a stable coin pegged to gold via an algo. so some of that language is still in there but i promise you can do so much more with it.

**market dynamics**
- growth and lack there of: you can control how many people flood in and how easy they are willing to panic sell
- hype: simulate outside hype and '2thamoon' attitude
- market regime shifts (bull/bear/neutral) that mess with your coins
- whale, retail, and other types of buyers behavior simulation
- organic user growth with adoption curves and churn mechanics
- liquidity depth modeling with real slippage calculation

the market dynamics are by far the hardest for me to sim accurately. i believe they are the most important. messing with the knobs on these are extremely interesting bc you can control how the outside market works and mess with the world as a whole. you can get postquadratic growth of purely users with diamond hands or vice versa.

**attacker lab** 
- configurable attacker profiles with different strategies (momentum traders, pump & dumpers, liquidity sappers, arb snipers)
- manual attack scheduling -- plan your own rug pulls and watch them unfold
- automated attacker algo that learns and adapts to market conditions
- flash crash simulation with recovery dynamics
- coordinated attack waves

these are soo fun to play with. you can give goals to the different attackers and watch as they optimize move by move for it. you can sim pump and dumps and see how well you can design an algo that will be able to recover quickly 

**the algorithm**
- algorithmic stability controls (circuit breakers, adaptive mint/burn, liquidity support)
- crash defense systems that deploy treasury resources when shit hits the fan
- supply adjustment algorithms with multiple regime modes (balanced, decay, hard cap, adaptive)
- treasury management with NAV tracking
- gas subsidy pools and fee rebate systems

okay this one might be more fun to mess with than the attacker stuff. The idea is having a central bank (that you can flush with cash via transaction fees and other mechanisms) that can use its resources to protect a goal you have (stablize prices, defend against attacks, maximize growth, etc). You have so many options here and finding the ideal setup to hit your goals is fuckin hard. super rewarding. Fwiw this is all optional and off by default -- 99% of coins do not do this irl

**token economics**
- customizable supply schedules with unlock mechanics
- transaction tax systems with burn vaults
- halving events and emission curves
- price-triggered supply adjustments
- inflationary vs deflationary regime switching
- supply hard caps and reversal mechanisms

these settings are _mostly_ in the market dynaics tabs. super interesting to mess with supply, it is incredibly important in terms of the end price and market cap. 

**policy modules**
- open market operations to stabilize price
- dynamic fee adjustments based on market conditions
- savings rate mechanisms for hodlers
- subsidies for people who hold and gas prices at a specific price  
- policy arbitrage flow simulation
- circuit breakers for extreme market moves

these are in the algo settings, they are just some specific knobs that allow you to tune the algo. i think stuff like allowing the treasury to subsidize gas cost when you are selling at the desired price is a really cool incentive and it works well

**visualization & analysis**
- real-time price charts with multiple overlay options
- supply dynamics tracking (circulating, locked, treasury, burned)
- liquidity depth visualization
- user growth and confidence metrics
- attacker buy and sell points so you can see what the algo was doing trying to reach its goal
- market regime indicators

so so so many charts -- arguably too many but thats neither here nor there -- they all let you visualize different factors of such a complex simulation


basically you can simulate anything from a legit stablecoin to a shitcoin rugpull and everything in between. i have so much fun setting goas in my head and then messing with knobs to try and do it. its really hard to make a bitcoin like coin, even if you cheese stuff like crazy user growth





## code base
fuck its so so bad god its just one file why did i make it one file. its 5000 lines all contained in app.py I gave it to codex under supervision at first however i wanted it to do all the mathy stuff. It kept producing extremely good outputs so i stopped looking at the codebase. around 1200 lines i stopped paying attention to the codebase all together. if you wanna make changes codex almost always can do it first try. claude sort of sucks at this. i tried one prompt with it to compare and it was very bad. 


## final thoughts
there is so much to add and there is for sure stuff that is inaccurate. this is all done by someone who doesn't know what makes crypto work (and the more i read into it, its becoming pretty apperent that seemingly nobody does) so there is probably (definitely) inaccuracies with it. i think you guys should try it tho

## legal and contributing
mit license; idgaf what you do with this

in terms of prs if you decide to submit one i will read it and i will probably merge it provided it isnt like insanely buggy or something
