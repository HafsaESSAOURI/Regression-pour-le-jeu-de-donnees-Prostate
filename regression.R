pros=read.table("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data")
print(pros)

#Varibles prédictives:
#lcavol: log(volume du cancer)
#lwaight: log(poids de la prostate)
#age
#lbph: log(quantité d'hyperplasie bénigne de la prostate)
#svi: invasion des véhicules séminales
#lcp: log(pénetration capsulaire)
#gleason: score de Gleason
#pgg45: pourcentage score de Gleason 4 ou 5
#lpsa (variable de réponse): log(antigène spécifique de la prostate)

#Conversion de la table en matrice
#matrix_pros = as.matrix(pros)
#print(matrix_pros)

#diviser notre dataset en train set et test set
condition <- pros$train == 1
train_set <- pros[condition, ]
test_set <- pros[!condition, ]
print(test_set)

#diviser train_set en train_X et train_Y
colonnes_a_exclure <- c("lpsa")
train_X <- train_set[, setdiff(colnames(train_set), colonnes_a_exclure)]
train_Y <- train_set$lpsa
train_Y <- as.vector(train_Y)
col_a_extraire <- c("train")
train_X <- train_X[, setdiff(colnames(train_X), col_a_extraire)]
train_X <- as.matrix(train_X)
#diviser test_set en test_X et test_Y
test_X <- test_set[, setdiff(colnames(test_set), colonnes_a_exclure)]
test_X <- test_X[, setdiff(colnames(test_X), col_a_extraire)]
test_X <- as.matrix(test_X)
test_Y <- test_set$lpsa
test_Y <- as.vector(test_Y)
#nbre d'obsérvations de l'ensemble d'entrainement
m_train = nrow(train_set)
print(m_train)
#nbre d'observations de l'ensemble de test
m_test = nrow(test_set)
print(m_test)
#nbre de variables prédictives
p = ncol(train_X)
print(p)
############# MCO ##################
MCO <-function(X,Y){
  sol<-solve(t(X)%*%X)%*%t(X)%*%Y
  return(sol)
}

Beta_MCO <- MCO(train_X, train_Y)
print(Beta_MCO)

#calculer l'erreur quadratique moyenne de prédiction sur les données de test
Y_hat = test_X %*% Beta_MCO
ecarts_carres <- (Y_hat - test_Y)^2
mse_MCO = mean(ecarts_carres)
print(mse_MCO) #0.5179699
####################################
####################################
############ Ridge #################
Ridge <- function(X, Y, lambda){
  p = ncol(X)
  sol <- solve(t(X)%*%X + diag(lambda, p, p))%*%t(X)%*%Y
  return(sol)
}
#choix de lambda à l'aide du cross validation
lambdas<- seq(from=0.001, to=50, by=0.001)
erreurLambda <- numeric(length(lambdas))
j=1
for(lamb in lambdas){
  mse =0
  
  for(i in 1:row(train_X)){
    X_training <- train_X[-i,]
    Y_training <- train_Y[-i]
    X_test <- train_X[i,]
    Y_test <- train_Y[i]
    Beta <- Ridge(X_training,Y_training,lamb)
    mse = mse + abs(Y_test- X_test %*% Beta)
  }
  erreurLambda[j] = mse/nrow(train_X)
  j = j+ 1
  
  
}
min = erreurLambda[1]
indiceLambda = 1
for(i in 2:length(erreurLambda)){
  if(erreurLambda[i]<min){
    indiceLambda = i
    min = erreurLambda[i]
  }
}

lambda <- lambdas[indiceLambda]
print(lambda) #0.192

Beta_Ridge <- Ridge(train_X, train_Y, lambda)
print(Beta_Ridge)

#calculer l'erreur quadratique moyenne de prédiction sur les données de test
Y_hat = test_X %*% Beta_Ridge
ecarts_carres <- (Y_hat - test_Y)^2
mse_Ridge = mean(ecarts_carres)
print(mse_Ridge)


####################################
####################################
############ Lasso #################
#calcul des coefficients lasso
coord_descLasso<-function(X,Y,lambda){
  beta1<-rep(0,dim(X)[2])
  beta2<-rep(1,dim(X)[2])
  while(sqrt(t(beta1-beta2)%*%(beta1-beta2))>0.0000001){
    
    if(dim(X)[2]==2)
    { beta2<-beta1
    for(j in 1:2)
    {Rj<-t(X[,j])%*%(Y-X[,-j]*beta1[-j])
    betaj<-Rj*max(1/(t(X[,j])%*%X[,j])-lambda/(2*abs(Rj)*(t(X[,j])%*%X[,j])),0)
    beta1[j]<-betaj}
    }
    else
    {  beta2<-beta1
    for(j in 1:dim(X)[2])
    {Rj<-t(X[,j])%*%(Y-X[,-j]%*%(beta1[-j]))
    betaj<-Rj*max(1/(t(X[,j])%*%X[,j])-lambda/(2*abs(Rj)*(t(X[,j])%*%X[,j])),0)
    beta1[j]<-betaj}}
    
  }
  return(beta1)
}



#choix de lambda à l'aide du cross validation
lambdas<- seq(from=0.001, to=5, by=0.001)
erreurLambda <- numeric(length(lambdas))
j=1
for(lamb in lambdas){
  mse =0
  
  for(i in 1:row(train_X)){
    X_training <- train_X[-i,]
    Y_training <- train_Y[-i]
    X_test <- train_X[i,]
    Y_test <- train_Y[i]
    Beta <- coord_descLasso(X_training,Y_training,lamb)
    mse = mse + abs(Y_test- X_test %*% Beta)
    
  }
  erreurLambda[j] = mse/nrow(train_X)
  j = j+ 1
  print(j)
  
}
min = erreurLambda[1]
indiceLambda = 1
for(i in 2:length(erreurLambda)){
  if(erreurLambda[i]<min){
    indiceLambda = i
    min = erreurLambda[i]
  }
}

lambda <- lambdas[indiceLambda]
print(lambda)
#lambda = 3.696

Beta_Lasso <- coord_descLasso(train_X, train_Y, lambda)
print(Beta_Lasso)

#calculer l'erreur quadratique moyenne de prédiction sur les données de test
Y_hat = test_X %*% Beta_Lasso
ecarts_carres <- (Y_hat - test_Y)^2
mse_Lasso = mean(ecarts_carres)
print(mse_Lasso)
####################################
####################################
############ Elastic net ###########
#calcul de l'estimateur Elastic Net à l'aide de la méthode de descente par coordonnées
coord_descElasticnet<-function(X,Y,lambda1,lambda2){
  beta1<-rep(0,dim(X)[2])
  beta2<-rep(1,dim(X)[2])
  while(sqrt(t(beta1-beta2)%*%(beta1-beta2))>0.0000001){
    
    if(dim(X)[2]==2)
    { beta2<-beta1
    for(j in 1:2)
    {Rj<-t(X[,j])%*%(Y-X[,-j]*beta1[-j])
    betaj<-(1/(1+(lambda2/(t(X[,j])%*%X[,j]))))*Rj*max(1/(t(X[,j])%*%X[,j])-lambda1/(2*abs(Rj)*(t(X[,j])%*%X[,j])),0)
    beta1[j]<-betaj}
    }
    else
    {  beta2<-beta1
    for(j in 1:dim(X)[2])
    {Rj<-t(X[,j])%*%(Y-X[,-j]%*%(beta1[-j]))
    betaj<-(1/(1+(lambda2/(t(X[,j])%*%X[,j]))))*Rj*max(1/(t(X[,j])%*%X[,j])-lambda1/(2*abs(Rj)*(t(X[,j])%*%X[,j])),0)
    beta1[j]<-betaj}}
    
  }
  return(beta1)
}

lambdas1<- seq(from=1, to=5, by=0.1)
lambdas2<- seq(from=1, to=5, by=0.1)
erreurLambdas <- matrix(nrow=length(lambdas1), ncol = length(lambdas2))
k=1
for(lamb1 in lambdas1){
  j=1
  for(lamb2 in lambdas2){
    mse =0
    
    for(i in 1:row(train_X)){
      X_training <- train_X[-i,]
      Y_training <- train_Y[-i]
      X_test <- train_X[i,]
      Y_test <- train_Y[i]
      Beta <- coord_descElasticnet(X_training,Y_training,lamb1, lamb2)
      mse = mse + abs(Y_test- X_test %*% Beta)
      
    }
    erreurLambdas[k,j] = mse/nrow(train_X)
    j = j+ 1
    print(j)
  }
  k = k+1
  print(k)
}

# Trouver la position du minimum
indice_min <- which.min(erreurLambdas)
lambda1_indice <- row(erreurLambdas)[indice_min]
lambda2_indice <- col(erreurLambdas)[indice_min]
print(lambdas1[lambda1_indice]) #4.4
print(lambdas2[lambda2_indice]) #1


Beta_ElasticNet <- coord_descElasticnet(train_X, train_Y, lambdas1[lambda1_indice], lambdas2[lambda2_indice])
print(Beta_ElasticNet)

#calculer l'erreur quadratique moyenne de prédiction sur les données de test
Y_hat = test_X %*% Beta_ElasticNet
ecarts_carres <- (Y_hat - test_Y)^2
mse_ElasticNet = mean(ecarts_carres)
print(mse_ElasticNet) #0.514182
######################################################################
######################################################################
############ Régression en compososantes principales##################
CP <- function(X, Y, k){
  #DVS de la matrice X
  decomposition_svd <- svd(X)
  valeurs_singulieres <- decomposition_svd$d
  U <- decomposition_svd$u
  V <- decomposition_svd$v
  sol <- V[, 1:k]%*%solve(t(diag(valeurs_singulieres[1:k],k,k))%*%diag(valeurs_singulieres[1:k],k,k))%*%t(diag(valeurs_singulieres[1:k],k,k))%*%t(U[, 1:k])%*%Y
  return(sol)
}
#choix de k à l'aide du cross validation
K<- seq(from=1, to=p, by=1)
erreurK <- numeric(length(K))
j=1
for(k in K){
  mse =0
  
  for(i in 1:nrow(train_X)){
    X_training <- train_X[-i,]
    Y_training <- train_Y[-i]
    X_test <- train_X[i,]
    Y_test <- train_Y[i]
    Beta <- CP(X_training,Y_training,k)
    mse = mse + abs(Y_test- X_test %*% Beta)
  }
  erreurK[j] = mse/nrow(train_X)
  j = j+ 1
  
  
}
min = erreurK[1]
indiceK = 1
for(i in 2:length(erreurK)){
  if(erreurK[i]<min){
    indiceK = i
    min = erreurK[i]
  }
}

k <- K[indiceK]
print(k)

Beta_CP <- CP(train_X, train_Y, k)
print(Beta_CP)

#calculer l'erreur quadratique moyenne de prédiction sur les données de test
Y_hat = test_X %*% Beta_CP
ecarts_carres <- (Y_hat - test_Y)^2
mse_CP = mean(ecarts_carres)
print(mse_CP)







