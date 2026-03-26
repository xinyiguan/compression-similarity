;; Start IDyOM
(start-idyom)

;; Import test data set into position 0
;; NOTE: This should only be run once
(db:import-data :krn "humdrum/" "Chorale melodies BWV255 and BWV340" 0)

;; Calculate normalised symetric compression distance using pitch viewpoint
(similarity:idyom-dissimilarity
  '(0) '(0)
  '(cpitch) '(cpitch)
  :output-path "output/"
  :symmetric t
  :normalised t)
