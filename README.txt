Thank you for being interested in my work!

This is some of the work I did for my Masters of Cognitive Science degree at
Carleton. The basic idea was to resolve the problem of contextual incoherence
in terms of the visual imagination. For example, most people when imagining a
scene based on the query 'mouse' will not imagine a computer mouse in a nest
with baby animal mice. Both Coherencer in Thesis.py and SemNet in SemNet.py
resolve this problem, but in different ways. I refer you to each of the files
in order to find the details of these differences.

If you use this code in your research, please include one or all of the
following citations:

Vertolli, M. O. & Davies, J. (2013). Visual imagination in context: Retrieving a coherent set of labels with Coherencer. In R. West & T. Stewart (eds.), Proceedings of the 12th International Conference on Cognitive Modeling, Ottawa: Carleton University. 
Vertolli, M. O. & Davies, J. (2014). Coherence in the visual imagination: Local hill search outperforms Thagard’s connectionist model, Proceedings of the 36th Annual Conference of the Cognitive Science Society. Quebec City, QC: Cognitive Science Society.
Vertolli, M. O., Breault, V., Ouellet, S., Somers, S., Gagné, J., & Davies, J. (2014). Theoretical assessment of the SOILIE model of the human imagination, Proceedings of the 36th Annual Conference of the Cognitive Science Society. Quebec City, QC: Cognitive Science Society.

In order to use each of the algorithms, do the following:

from SemNet import SemNet as S
s = S() #if using the base path

from Thesis import Coherencer as C, Comparer as Cm
c = C()
cm = Cm()

results1 = s.askCycle(num=8372)
results2 = c.askCycle(num=8372)

cm.test(results1)
cm.test(results2)


The output from each test will be the total number of images in the database
that contain the returned set of labels for a given query.

If you want each label to have a consistent index across both algorithms do
the following before calculating results1 and 2:

s.buildTermsToProc(num=8372)
c.termsToProc = [term for term in s.termsToProc]
