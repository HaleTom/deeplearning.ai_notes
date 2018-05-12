# Week 2

## Learning Objectives
* Understand what multi-task learning and transfer learning are
* Recognize bias, variance and data-mismatch by looking at the performances of your algorithm on train/dev/test sets

## Error Analysis

If performance is not yet human level, then manually examining mistakes can give insights as to what to do next. This process is called *error analysis*.

Example: Classifier with 10% error is misclassifying some dogs as cats.

What path to take?
* Collect more dog pictures?
* Design features specific to dogs?

It could take months to work on the dog problem, and there may not be sufficient benefit.

Error analysis can tell whether or not it is worth the effort.

Error Analysis:
1. Get about 100 mislabeled Dev set examples
1. Manually count up how many are dogs

Suppose 5 mislabelled images are dogs. This means that even if the dog problem were solved, it would only affect 5 out of 100 misclassifications.

The best error reduction based on dogs would be from 10% down to 9.5%, or a 5% relative decrease in error.

This 5% gives a ceiling or upper bound on how much performance can be increased by working on dog misclassifications.

It may well be worth while working on the largest case of misclassifications reather than dogs.

Sometimes in ML it is disparaged to hand engineer things, but if building applied systems, this simple counting procedure can save a lot of time deciding what is the most important problem to focus on.

### Error analysis on multiple ideas in parallel

Ideas for improving cat detection:

* Fix dogs being recognised as cats
* Fix big cats (lions, etc) being misrecognised
* Improve performance on blurry images

Create a spreadsheet with columns:

![wk2-error-analysis.png](wk2-error-analysis.png)

Count up how many fall into each class to get a percentage. Make up new classes of error if useful if there seems to be more commonality.

The above analysis will take a maximum of a couple of hours, but could save months of working on something that may only make a minor difference.

## Cleaning up incorrectly labelled data

Andrew uses "mislabelled examples" for where prediction $\hat y \ne y$, but "incorrectly labelled examples" where the data set label $y$ is wrong.

### Training set errors

DL algorithms are relatively robust to random or near-random training data set errors, eg keyboard input errors in labelling.

There's no harm in fixing incorrect labels, but things may be ok even if you don't as long as the total data set size is big enough and the number of errors is not too high.

Caveat: DL algorithms are less robust to systematic errors (eg most small white dogs are incorrectly labelled cats).

### Dev / Test errors

Add a column to the error analysis spreadsheet for "incorrectly labelled".

![dev-test-incorrectly-labelled.png](dev-test-incorrectly-labelled.png)

Advice: Only relabel if it makes a significant difference to the ability to evaluate models on the Dev set.

Look at the:

* Overall Dev set error
* Overall error due to incorrect labels
* Overall error due to other causes

If the "other causes" is much higher, then better accuracy could be achieved by working on those things first.

Remember the purpose of the dev set is to rank performance of algorithms.

Assume classifier A has an error of 2.1% and classifier B has an error of 1.9%, with 0.6% error coming from incorrect labels. In this case the incorrect labels will obfuscate the performance difference.
