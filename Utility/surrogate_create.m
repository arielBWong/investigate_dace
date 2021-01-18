function [krg, krgc] = surrogate_create(trainx, trainy, trainc, normhn, daceflag)

trainy_norm = normhn(trainy);
krg = surrogate_train(trainx, trainy_norm, daceflag);
krgc = surrogate_train(trainx, trainc, daceflag);


end