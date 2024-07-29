Content:

FCNN_unified.ipynb - main file with training script
checks_unified.ipynb - all the validation and analysis for this projects 

Abstact:

Artificial Intelligence in the task of prediction of differential cross sections and structural functions of pion-proton production in the resonance region. 

Golda Andrey Vasilyevich
Postgraduate student
Lomonosov Moscow State University, 
Faculty of Physics, Moscow, Russia
E-mail: golda.av15@physics.msu.ru

Every year methods of artificial intelligence and neural networks in particular become more and more powerful tools for studying various fields of science. Elementary particle physics is not an exception. Artificial intelligence algorithms can already be used to build data generators [1], suppress noise and background processes in data, determine particle tracks [2] etc. Particle physics researchers work with detectors and, consequently, with large amounts of data that allow efficient construction of machine learning models. The variety of detector types allows the identification of particle types, their momenta and energies, which subsequently allows the extraction of the reaction scattering cross section. The reaction cross section is the main quantity used by physicists who study collisions of objects in the micro world. It is the reaction cross sections that determine the probability of reactions occurring. 
In this paper we study the methods of artificial intelligence in the task of predicting differential cross sections of reactions in the processes of the production of positively charged pions under the action of electrons. When considering the reaction of pion production:

e- + p -> e- + n + pi+ -> e- + n + u + mu+

data set was generated to predict the reaction cross section in different regions of phase space for different electron beam energies. The algorithm to be trained is a fully-connected neural network with X hidden layers, which was trained with a loss function that did not include any theoretical knowledge of the process. In addition to standard validation procedures for the regression problem, this paper also presents physically based comparisons of unpolarised structural response functions, which are calculated from the predicted cross-sectional values. Based on this algorithm, it is possible to interpolate and extrapolate both cross section values and structure function values in different regions of phase space, which is of great value in investigating the nature of strong interaction. 

[1] T. Alghamdi, Y. Alanazi, M. Battaglieri, Ł. Bibrzycki, A. V. Golda, A. N. Hiller Blin, E. L. Isupov, Y. Li, L. Marsicano, W. Melnitchouk, V. I. Mokeev, G. Montaña, A. Pilloni, N. Sato, A. P. Szczepaniak, and T. Vittorini (2023). Toward a generative modeling analysis of CLAS exclusive 2π photoproduction. Phys. Rev. D 108, 094030
[2]	P. Thomadakis, A. Angelopoulos, G. Gavalian, N. Chrisochoides (2022). Using Machine Learning for Particle Track Identification in the CLAS12 Detector. Computer Physics Communications 276, 152
