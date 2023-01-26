ZONE=us-central1-a
PROJECT=brain-complex-systems-xgcp

for i in $(seq -w 00 19); do
  INSTANCE_NAME=ct-collect-$i
  gcloud compute instances create "${INSTANCE_NAME}" \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --machine-type=n1-standard-96 \
    --image-family=tf-ent-latest-cpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True,proxy-mode=service_account" \
    --tags=deeplearning-vm,ssh-tunnel-iap \
    --scopes=cloud-platform,storage-full,cloud-source-repos
done
