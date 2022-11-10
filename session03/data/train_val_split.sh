cd catsDogs

mkdir validation_set
mkdir validation_set/dogs
mkdir validation_set/cats
cd training_set/cats
for i in `seq 3200 4000`;
  do
    mv cat.${i}.jpg ../../validation_set/cats;
done

cd ../dogs
for i in `seq 3200 4000`;
  do
    mv dog.${i}.jpg ../../validation_set/dogs;
done