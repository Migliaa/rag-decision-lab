# Lettura manuale del contesto — tre domande, tre regimi di recupero

Generato da `01_lettura_manuale.py`. Configurazione di recupero: fusione RRF
multi-formulazione, riordino con cross-encoder sui primi 30, primi 10 passaggi
nel contesto. Le annotazioni a mano vanno scritte sotto ogni domanda, nei punti segnati
con `>`.

Conteggio dei regimi su tutte le 180 domande: **A** 64 domande, **B** 23 domande, **C** 21 domande, restanti 72 con oro fra i primi dieci ma non al primo posto.

---

## Regime A — oro al primo posto — contesto nella condizione migliore

**Domanda (id `1c74814752c3f9d4ed0b99c16e1ed192<::>2`)**

- ultimo turno, cioe' cio' che l'utente scrive davvero: *Do employers need to report their employee's salaries and withhold money for income tax purposes?*
- riscritta da annotatore umano, cioe' cio' che il recupero riceve qui: *Do employers need to report their employees' salaries and withhold money for income tax purposes?*

**Fonti annotate come corrette**: 3. Posizione finale di ciascuna: fuori dai 60 candidati, 2, 1.

> *Che aspetto avrebbe una risposta corretta a questa domanda:*

> *Come si riconoscerebbe qui un fallimento di generazione (contesto sufficiente, risposta sbagliata):*

### Le fonti annotate

**[oro] 245451-0-792** — posizione finale: 1

Many countries have employers report their employees' salaries and withhold some money for income tax purposes (it's called “pay as you earn”, “withholding taxes” or taxing “at the source”). Often the system is designed in such a way that most people actually pay too much and can get money back at the end of the year. In that case, the salary you receive can certainly be considered a “net salary”. Depending on the tax system, individuals might need to file a separate tax returns to report any other income (investments, rents, whatever) or benefit from tax incentives but it can also be optional. If you live in such a country and your situation is simple enough that there are no applicable deductibles, you might simply choose to forgo it and let your employer take care of everything.

**[oro] 462036-6404-8453** — posizione finale: fuori dai candidati

When you begin working, an employer will usually have you fill out a tax form such as a W-4 Employee's Withholding Allowance Certificate so that your employer can withhold the correct federal income tax from your wages. If they don't, then it is your responsibility to calculate and file your own income taxes (if you are self-employed, an independent contractor or paid under the table). If your employer is reputable, they will send you additional information (generally in February) you need to properly file your taxes prior to April 15th (the IRS tax deadline for most people). This additional information will likely be some variation of a W-2 Wage and Tax Statement or possibly a Form 1099-MISC. Do I have to worry about money in my bank account? Unless you have a significant amount in your bank savings account earning interest (see ""Should I save for the future?"" below), you won't have to pay any sort of tax on money in your bank account. If you do earn enough taxable interest, the bank will send you the proper forms to file your taxes. How do I file taxes? While it won't apply till next year, you will likely be able to fill out a Form 1040EZ Income Tax Return for Single and Joint Filers With No Dependents, as long as you don't have any kids in the meantime. ;-) You will either mail in the paper form (available at your local IRS office, post office, public library, etc.) or [...]

**[oro] 597053-0-714** — posizione finale: 2

Yes. W4 determines how much your employer will withhold from your wages. Leaving everything at default would mean that your salary is your only taxable income, and you only take default deductions. Your employee will calculate your tax withholding based on that. But, if your salary is >200k, I assume that you have other income (investment/capital gains, interest on your bank account), which you will have to pay taxes on. You're probably going to have some deductible expenses (business/partnership expenses, mortgage interest, donations, college funds etc) as well. So it is very likely, unless you're really not smart about money, that you have more to do with your taxes than just the employers' withholding.

### Il contesto che il generatore riceverebbe

**1. 245451-0-792** **(annotata corretta)**

Many countries have employers report their employees' salaries and withhold some money for income tax purposes (it's called “pay as you earn”, “withholding taxes” or taxing “at the source”). Often the system is designed in such a way that most people actually pay too much and can get money back at the end of the year. In that case, the salary you receive can certainly be considered a “net salary”. Depending on the tax system, individuals might need to file a separate tax returns to report any other income (investments, rents, whatever) or benefit from tax incentives but it can also be optional. If you live in such a country and your situation is simple enough that there are no applicable deductibles, you might simply choose to forgo it and let your employer take care of everything.

**2. 597053-0-714** **(annotata corretta)**

Yes. W4 determines how much your employer will withhold from your wages. Leaving everything at default would mean that your salary is your only taxable income, and you only take default deductions. Your employee will calculate your tax withholding based on that. But, if your salary is >200k, I assume that you have other income (investment/capital gains, interest on your bank account), which you will have to pay taxes on. You're probably going to have some deductible expenses (business/partnership expenses, mortgage interest, donations, college funds etc) as well. So it is very likely, unless you're really not smart about money, that you have more to do with your taxes than just the employers' withholding.

**3. 470898-0-1766**

To clarify that legality of this (for those that question it), this is directly from IRS Publication 926 (2014) (for household employees): If you prefer to pay your employee's social security and Medicare taxes from your own funds, do not withhold them from your employee's wages. The social security and Medicare taxes you pay to cover your employee's share must be included in the employee's wages for income tax purposes. However, they are not counted as social security and Medicare wages or as federal unemployment (FUTA) wages. I am sorry this does not answer your question entirely, but it does verify that you can do this. UPDATE: I have finally found a direct answer to your question! I found it here: http://www.irs.gov/instructions/i1040sh/ar01.html Form W-2 and Form W-3 If you file one or more Forms W-2, you must also file Form W-3. You must report both cash and noncash wages in box 1, as well as tips and other compensation. The completed Forms W-2 and W-3 in the example (in these instructions) show how the entries are made. For detailed information on preparing these forms, see the General Instructions for Forms W-2 and W-3. Employee's portion of taxes paid by employer. If you paid all of your employee's share of social security and Medicare taxes, without deducting the amounts from the employee's pay, the employee's wages are increased by the amount of that tax for income [...]

**4. 386264-0-1024**

In general no, if you just have one employer and work there with the same salary for the whole year. Typically an employer does tax withholding by extrapolating your monthly income to the entire year and withholding the right amount so that at the end, what is withheld is what you owe. It's not a surprise to them when your income crosses a tax bracket threshold, because they knew how much they were paying you and knew when you would cross into another bracket, so they factored that in. If you have multiple jobs or only worked for part of the year, or if your income varied from month to month (e.g., you got a raise) there could be a discrepancy between what is withheld and what you owe, because each employer only knows about what it's paying you, not what money you may have earned from other sources. (Even here, though, the discrepancy wouldn't be due to the tax brackets per se.) You can adjust your withholdings on form W-4 if needed, to tell the employer to withhold more or less than they otherwise would.

**5. 549870-0-500**

"You are on the right track, for tax purposes its all ordinary income at the end of 2016. If the free lance ""employer"" will withhold fed,state and local tax, then that takes care of your estimated tax. If they can't or won't, you will need to make those estimates and make payments quarterly for the fed and state tax at your projected tax liability. Or, you can bump up withholding by your day job employer and cover your expected tax liability at year end without making estimated tax payments."

**6. 388713-2953-3790**

Most wage-earners don't notice this because it happens automatically through payroll withholding, but as self-employed you are responsible for making sufficient and timely estimated payments, and will owe a penalty if you don't. However, since this is your first year you may have a 'safe harbor'; if you also have income from an employer (reported on W-2, with withholding) and that withholding is sufficent to pay last year's tax, then you are exempt from the 'underpayment' penalty for this year. If you elect corporation status then the corporation (which is really just you) must always make timely payments of withheld amounts, according to one of several different schedules that may apply depending on the amounts; I believe it also must make estimated payments for its own liability, if any, but I'm not familiar with that part.

**7. 466213-0-1057**

"You file taxes as usual. W2 is a form given to you, you don't need to fill it. Similarly, 1099. Both report moneys paid to you by your employers. W2 is for actual employer (the one where you're on the payroll), 1099 is for contractors (where you invoice the entity you provide services to and get paid per contract). You need to look at form 1040 and its instructions as to how exactly to fill it. That would be the annual tax return. It has various schedules (A, B, C, D, E, F, H, etc) which you should familiarize yourself with, and various additional forms that you attach to it. If you're self employed, you're expected to make quarterly estimate payments, but if you're a salaried employee you can instruct your employer to withhold the amounts you expect to owe for taxes from your salary, instead. If you're using a tax preparation software (like TurboTax or TaxAct), it will ""interview"" you to get all the needed information and provide you with the forms filled accordingly. Alternatively you can pay someone to prepare the tax return for you."

**8. 99434-0-2012**

"I have an indirect answer for you. It is not a numeric answer but it is a procedure. The challenge with paying taxes for an employee besides their share of Social security and Medicare is that you have no idea what their state and Federal taxes are. Are they married, single, head of household? Is this their entire families income, or is it extra money to make ends meet? What about state taxes? It looks like you will need a W-4 from them. As you know the IRS Tax topic 756 has all the info you need. Federal Income Tax Withholding You are not required to withhold federal income tax from wages you pay to a household employee. However, if your employee asks you to withhold federal income tax and you agree, you will need a completed Form W-4 (PDF), Employee's Withholding Allowance Certificate from your employee. See Publication 15, (Circular E), Employer's Tax Guide, which has tax withholding tables that are updated each year. Form W-2, Wage and Tax Statement If you must withhold and pay Social Security and Medicare taxes, or if you withhold federal income tax, you will need to complete Form W-2 (PDF), Wage and Tax Statement, for each employee. You will also need a Form W-3 (PDF), Transmittal of Wage and Tax Statement. See ""What Forms Must You File?"" in Publication 926 (PDF) for information on when and where to furnish and file these forms. To complete Form W-2 you will need an [...]

**9. 361954-0-768**

Everyone pays their personal income tax with funds from their employer; some of it through withholding, and the rest through the balance due at the time of filing. All that is happening here is that the company is calculating your personal tax return for you, and fiddling retroactively with the gross salary to yield a specific after-tax salary. One problem is that there is a lot of information I put in my return to earn deductions, that I would not care to tell my employer. The system would also appear to be contrary to public policy. Governments create tax deductions to give a larger income to those with socially acceptable expenses: health care, dependents, etc. The system you describe would give employees with such deductions a lower gross salary.

**10. 557603-0-1971**

"Employers withhold at rates specified in Circular E issued by the IR. You can request that additional money be withheld (not an issue here) or you can have reduced withholding by claiming additional allowances on a W-4 (i.e., more than just for you and spouse and dependents) if you believe that this will result in withholding that will more closely match the tax due. (Note added in edit):Page 2 of the W-4 form has worksheets that can be used to figure out how many additional allowances to request. Also, I wonder if your withholding will be 37% or final tax bill be 26% of your adjusted gross income. The tax brackets are the tax on marginal income. If you are in the 28% tax bracket, you owe 28 cents in tax for each additional dollar of income, not 28 cents in tax for every dollar of income. Your overall tax might well be less than 20% of your income. As a specific example, in 2011 a married taxpayer filing jointly would be in the (highest) 35% tax bracket if the taxable income was $379,150 or more (marginal tax rate of 35% is applicable to every dollar more than $379,150) but the tax on $379,150 itself works out to be $102,574 or 27.05% of the taxable income. So if you do expect to be earning around $350K or more in salary between now and December 31 to hit that 26% that you expect you will owe, you might want to consider paying a tax accountant for advice on how to fill out [...]

> *Dei dieci passaggi qui sopra, quanti rispondono davvero alla domanda (oltre a quelli annotati)? Le annotazioni di FiQA sono incomplete, e questo conteggio a mano e' l'unico modo di sapere quanto lo sono su questa domanda:*

> *Il contesto basta per rispondere?*

---

## Regime B — oro fra i candidati ma fuori dai primi dieci — il riordino lo ha perso

**Domanda (id `1c74814752c3f9d4ed0b99c16e1ed192<::>9`)**

- ultimo turno, cioe' cio' che l'utente scrive davvero: *If I lose my job*
- riscritta da annotatore umano, cioe' cio' che il recupero riceve qui: *What happens to my income tax if I lose my job?*

**Fonti annotate come corrette**: 3. Posizione finale di ciascuna: 49, fuori dai 60 candidati, fuori dai 60 candidati.

> *Che aspetto avrebbe una risposta corretta a questa domanda:*

> *Come si riconoscerebbe qui un fallimento di generazione (contesto sufficiente, risposta sbagliata):*

### Le fonti annotate

**[oro] 229134-0-1498** — posizione finale: fuori dai candidati

"To receive unemployment benefits, you must be registered with an employment agency and be actively seeking work, and be willing to accept work should it be offered to you. As a full-time Cornell University student, as you describe yourself, this does not seem a likely scenario. Also, you need to have established a state of residence. It is not clear to me that you have done so, given your travel between the South Pacific, Ithaca, New York, North Carolina etc. You should check with your local state unemployment office in New York State, or perhaps North Carolina, although I don't know if you satisfy residency requirements in either state. They will be able to confirm however. Are your parents claiming you as a dependent on their federal income taxes? If so, I do not believe that you will be able to file for unemployment benefits, regardless of your student status at Cornell University. One more issue to consider: Have you filed tax returns for the income you received from your television production work? I am uncertain of the amount, as you said that you worked for two 4-month intervals making $10,000. That implies $20,000 of earnings over two years. Yet your bullet point number 7 states that you made ""10 grand each of the four months"". If that means that you made $10,000 per month for four months, then you earned $40,000 per summer, for a total of $80,000 for two summer's [...]

**[oro] 307120-0-2515** — posizione finale: 49

"Unemployment insurance provides a temporary safety net to workers who lose their jobs by replacing a portion of their salary for certain periods. Each state administers its own unemployment insurance program so some rules may vary from state to state. To receive unemployment insurance payments, you must have lost your job through no fault of your own. If you quit your job or lost it because of poor performance or another justifiable reason, you are not eligible for unemployment insurance benefits. State unemployment insurance programs require claimants to have worked sufficiently before they can claim benefits. As soon as you apply for unemployment insurance, an agency with the state in which you live will verify that you were a victim of a layoff by contacting your previous employer and making sure you lost your job due to lack of work and not an action within your control. After the state verifies you were indeed the victim of a layoff, your weekly payment is calculated. Your payment will be a percentage of what you made in your previous job, generally between 20 percent and 50 percent, depending on your state. Unemployment insurance replaces only a portion of your previous pay because it is intended to pay only for the essentials of living such as food and utilities until you find new employment. Before you begin receiving benefits, you must complete a waiting period of [...]

**[oro] 481896-0-530** — posizione finale: fuori dai candidati

Unemployment is meant for people who were laid off, not terminated. Also, the employee who claimed unemployment did so even after they had worked for another company for several months between working for me and making the claim. If I actually laid someone off I would be the first to personally help them fill out paperwork for unemployment benefits. If those benefits go to those who shouldn't receive them it hurts everyone. That 3% increase in unemployment insurance is 3% that isn't going into paying my actual employees.

### Il contesto che il generatore riceverebbe

**1. 293134-0-650**

Bad idea. If you lose your job and need to pay medical expenses, you can withdraw penalty-free. If you lose your job and just withdraw, you will have lower income and lower tax, though you will pay a penalty. If you don't lose your job today, consider your 401(k) an additional protection, just in case you do lose it tomorrow. Just pick the least risky investment option and relax. Besides, it will diversify your investment and protect you from your own investment decisions. If you are so ready to take a 10% loss, you are likely not so skilled in this area, so it's good to have a backup plan. But of course, do contribute 3% so you get a match.

**2. 280538-0-470**

In some circumstances losses from self-employment can be offset against total income and/or capital gains. If this applies to you may be able to claim back some of the tax taken by PAYE from your day job. You can also to some extent carry the loss backwards into previous tax years or forward into the next one if you can't use it fully this year. HMRC have some information available on the current rules: When you can claim losses You can claim: But You can’t claim:

**3. 321387-0-98**

If you leave your job (or lose it) the loan is due on separation. You'll pay tax and a10% penalty.

**4. 475668-0-898**

For reference see this article. This article does an okay job of explaining why, but it could be better. To expand on point #4 if you lose your job, you will be forced to repay the loan in 90 days. If do not pay it back in time, you will be hit with your highest marginal tax rate and a 10% penalty. How does borrowing money at 40% interest sound? Why do you have credit card debt? I'll give you the loving answer: bad behavior. The longer you hold this debt the more indicative it is about out of control behavior. To remedy this I would recommend the following: While you are behaving like most people (normal); most people are broke. Congratulations on having the desire to not be broke. Do you now have the courage to change? Having that courage could mean generational wealth building and freedom from debt. As a reformed overspender it has meant exactly that for me and my family.

**5. 14142-0-352**

So we can lower your taxes, although at a cost of say 4 million federal employees. Now the fun begins... your market is flooded with new employees, so many people applying there isn't a reason to keep you at your pay. It's 2008 again, you lost your job, your $30 an hour job now pays $17, and you're unemployed, but at least your tax bracket went down.

**6. 291847-0-237**

Unfortunately, no. Think about the numbers. If you work for me, and I pay you $1000, you owe tax on $1000. If you still work, but I don't pay you, you have no tax due, but there's no benefit for you to collect for my stealing your time.

**7. 116009-0-588**

"Nothing ""happens"" to it. It works the same way regardless of whether you are a U.S. citizen or resident or not. Taxes and penalties work the same way on withdrawal. That said, if you are not in the U.S. and don't have any income in the U.S. in a particular year in the future, you can take advantage of the fact that your U.S. tax that year will probably be zero. Then, if you withdraw a little bit, even if they count as taxable income, your U.S. income will still be so low that it may be under your personal exemption, or if not at least it will be taxed in the lowest tax bracket."

**8. 52978-0-916**

You already received a tax deduction, more or less, as you didn't pay tax on this income. Beyond that, no; the money is lost if you don't spend it. See this explanation for example, among many others; it's specific to FSA, but they work generally the same way. To go into a bit more detail, read the IRS publication on the subject; basically, what's happening here is that you're receiving a fringe benefit, rather than salary. So yes, you agreed to voluntarily reduce your salary by $255 or whatever per month in exchange for funds in this account. As such, they're nontaxable, which both your employer and you find helpful; but the downside is it's not really your money - it's a fringe benefit. Nice that it's tax free, and dissimilar to a medical or dependent care FSA, it doesn't have an expiration date; but it does go away when you leave your employer, and you don't get it back. It's money you never had.

**9. 532888-0-1437**

"Short answer: Yes. For Federal income tax purposes, you are taxed on your total income, adding up positives and negatives. If business A made, say, $100,000 while business B lost $20,000, then your total income is $80,000, and that's what you'll be taxed on. As @littleadv says, of course any business losses you claim must qualify as business losses under IRS rules. And yes, there are special rules about losses that the IRS considers ""passive"". If you have wage income in addition to business income, business losses don't offset wage income for social security and medicare tax purposes. You can't get a refund of the social security tax deducted from your paycheck. I don't know if this is relevant to you, but: If you have businesses in different states, each is taxed by that state. For example I have two tiny side businesses, one in Michigan and one in Ohio. Last year the Michigan business made money while the Ohio business lost money. So my federal income was Michigan minus Ohio. My Ohio income was negative so I owed no Ohio income tax. But I couldn't subtract my Ohio losses from my Michigan income for Michigan income tax purposes. Thus, having, say, $10,000 income in Michigan and $10,000 in Ohio would result in lower taxes than $30,000 income in Michigan and a $10,000 loss in Ohio, even though the total income in both cases is the same. And this would be true even if the tax [...]

**10. 393725-0-213**

Yes but the employee losing that job can be devastating because as someone who works for minimum wage (in college and living at home right now so my cost of living is low) but without that I would have no savings.

> *Dei dieci passaggi qui sopra, quanti rispondono davvero alla domanda (oltre a quelli annotati)? Le annotazioni di FiQA sono incomplete, e questo conteggio a mano e' l'unico modo di sapere quanto lo sono su questa domanda:*

> *Il contesto basta per rispondere?*

---

## Regime C — nessun oro fra i 60 candidati — il recupero ha fallito a monte

**Domanda (id `1c74814752c3f9d4ed0b99c16e1ed192<::>3`)**

- ultimo turno, cioe' cio' che l'utente scrive davvero: *In that case, could you please explain why I still have to pay income tax?*
- riscritta da annotatore umano, cioe' cio' che il recupero riceve qui: *Why do I still have to pay income tax?*

**Fonti annotate come corrette**: 2. Posizione finale di ciascuna: fuori dai 60 candidati, fuori dai 60 candidati.

> *Che aspetto avrebbe una risposta corretta a questa domanda:*

> *Come si riconoscerebbe qui un fallimento di generazione (contesto sufficiente, risposta sbagliata):*

### Le fonti annotate

**[oro] 41509-0-965** — posizione finale: fuori dai candidati

You need to report the income from any work as income, regardless of if you invest it, spend it, or put it in your mattress (ignoring tax advantaged accounts like 401ks). You then also need to report any realized gains or losses from non-tax advantaged accounts, as well as any dividends received. Gains and losses are realized when you actually sell, and is the difference between the price you bought for, and the price you sold for. Gains are taxed at the capital gains rate, either short-term or long-term depending on how long you owned the stock. The tax system is complex, and these are just the general rules. There are lots of complications and special situations, some things are different depending on how much you make, etc. The IRS has all of the forms and rules online. You might also consider having a professional do you taxes the first time, just to ensure that they are done correctly. You can then use that as an example in future years.

**[oro] 280081-0-1578** — posizione finale: fuori dai candidati

"In the U.S., virtually all salaries are expressed as ""gross salaries"", which are before the taxes that the individual must pay on their income. The numbers shown in the links are almost certainly gross salary figures. However, the ""gross salary"" is not the entire ""total compensation"" number, which is the total value of all compensation and benefits that the employee receives for his work. Total compensation includes not only salary and bonuses, but the cost or value of any employer-paid healthcare, retirement, company car, expense account, stock options, and other valuable goods or services. That's still not the total amount of money the company has to pay to have you; there are employer-paid payroll taxes totaling 6.2% of your gross salary, plus practical costs like the cost of your computer, cubicle or office furniture, and the portion of utility costs that keep you well-lit, clean and comfortable. This complete number is called ""total employee cost"", and the general rule of thumb is that it's double your gross cash compensation (salary + bonuses). Lastly, $100k in California isn't worth as much, in real terms, as $100k in other parts of the U.S. The cost of living in California, especially in Silicon Valley where the majority of the people who make six figures by being C++ programmers are located, is ridiculously expensive. There are other tech hubs in the U.S., [...]

### Il contesto che il generatore riceverebbe

**1. 253541-0-739**

I think we should re-think about paying taxes. If they don't have to pay income tax. It is just one way: to minimize the revenue and maximize their expense. As you know, profit (or loss) = revenue - expense. If they want to reduce their income tax. They have to do this. We can't say anything about the income tax that G.E must pay. They maybe have some rights to encourage their business by being decreased some kinds of taxes. It's very normal. If you have some knowledge about tax and accounting, audits, You will have a clearly point about it! I'm not native speaker. I might not understand your tax rules. But they have all general standards all over the world. If I use wrongly grammar or misunderstanding, pls forgive me. Thank you!

**2. 37508-0-1306**

I know that if you make more, you pay more, but do those who have more, not make more, pay higher income tax? In general, no. In most locales, income tax is based on income, not on wealth. I am retired. I have little income but a fair amount of wealth. I play very little income tax. (But I do pay other kinds of taxes.) Here's a scenario. 2 people of average wealth with similar situations have the same job with equal pay. After 5 years, their situations haven't changed and they still earn equal pay, but now one has $40,000 in their account and the other $9,000. Does one now pay higher income tax because he has more in his account or does he pay the same because he makes the same? In most locales, you pay income tax on everything that is counted as income. Your salary is income. In some cases, earned interest is income. But aside from the earned interest from your bank accounts, neither the $40,000 nor the $9,000 is income. Your huge mansion isn't income. Your expensive car isn't income. The huge amount of land you own isn't income. The pricey artwork on your walls isn't income. You don't pay income tax on any of these, but your local may impose other taxes on these (such as property tax, etc.) [Note: consult the tax laws of your specific locale if you want to know details.]

**3. 259989-0-215**

You only have to pay income tax on a tax deferred account (like a 401k) when you withdraw money from it. You might only need $3K to live on a month, or less, so you only have to pay the taxes at that time I believe.

**4. 369000-0-374**

&gt;But you also said that's not why we pay taxes No, I didn't. &gt;the fact that they're using the infrastructure isn't a good enough reason to pay taxes. It's not. If everyone who used infrastructure paid income taxes than every person would pay some amount of income tax. Clearly just using infrastructure isn't considered a good enough reason to pay taxes now is it?

**5. 404429-0-553**

I'm not missing the point. Canada will still charge you/a corporation income taxes on worldwide income so long as you are resident in Canada. If you are incorporated in Canada but resident elsewhere, you are only subject to tax on Canadian-sourced income. In the US, where you are incorporated is the method of determining liability. Why is one method of determining jurisdiction correct and not the other? If you are a corporation residing in Canada, you still pay Canadian taxes on worldwide income, even if that income is sourced in another country.

**6. 446117-1556-2665**

Whether you have to make such payments depends on how much tax you owe and whether you've paid it by other means. You may need to pay estimated tax even if you're not self-employed, although this would be unusual. (It could happen, for instance, if you realized large capital gains over the year.) You also may be self-employed but not need to pay estimated tax (if, for instance, you also have a W-2 job and you reduce your withholding allowances to have extra tax withheld). That said, if you earn significant income from self-employment, you'll likely have to make estimated tax payments. These are prepayments of the income tax and Social Security/Medicare taxes you accrue based on your self-employment income. As Pete B. mentioned in his answer, a possible reason that your estiamtes are low is because some taxes have already been withheld from the paychecks you received so far during the year (while you were an employee). These represent tax payments you've already made; you don't need to pay that money a second time, but you may need to make estimated tax payments for your income going forward."

**7. 475630-0-459**

Uh...no, you didn't. &gt;I wish I You're talking about you, here, not businesses or corporations. &gt;didn't have to pay taxes if I had less in the bank at the end of the year than at the beginning. You're talking about not having any money and not having to pay tax. What *I* said was that a corporation can very well have *no* money and have to pay tax. A corporation's level of cash has no more bearing on its liability to pay taxes than does yours.

**8. 569188-0-660**

I don't see the connection between the investment capital and the income you receive from it, and why one part should be free from taxation just because the other was taxed. I mean with that logic everything I do with my after-tax money is hopefully providing me some form of anticipated benefit, so I guess I shouldn't have to pay taxes on any part of my income that i spend or put at risk... why is making a capital investment any different in this regard than paying for education, buying a tangible asset (like a home), buying food to keep me alive so i can continue to work, etc? Basically, you're going to have to explain why that makes sense to you.

**9. 355369-0-229**

It means you must pay federal (and possibly state) tax on any income you produce in America -- including Internet and mail-order sales. Tax treaties may keep you from having to pay tax on it again in your own country, or may not.

**10. 20988-0-350**

I'll point out that you don't actually have to pay your income taxes on time, just make sure to file on time. You will be charged interest (currently 5% ann., compounded daily). This is no big deal, I know plenty of people who do this. Note that if they pay you interest you have to report it on your taxes, but if you pay them it's not deductible.

> *Dei dieci passaggi qui sopra, quanti rispondono davvero alla domanda (oltre a quelli annotati)? Le annotazioni di FiQA sono incomplete, e questo conteggio a mano e' l'unico modo di sapere quanto lo sono su questa domanda:*

> *Il contesto basta per rispondere?*
