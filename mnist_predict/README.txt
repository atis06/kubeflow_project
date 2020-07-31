! pip show kfp

if not 
! pip3 install kfp --user
! pip3 install azure=="4.0.0" --user
! pip3 install kubeflow --user --no-cache

! kubectl apply -f config/minio_secret.yaml