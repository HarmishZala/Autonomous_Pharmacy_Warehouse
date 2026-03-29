;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; COMP0224 Coursework 2 — Task B: Patient Order Problem
;; Student ID: 23013383
;;
;; Scenario: A patient order requires insulin, amoxicillin, and ibuprofen.
;; Three robots operate in the warehouse:
;;   - picker1       : Picker arm stationed at the picking station
;;   - transporter1  : Mobile transporter, starts at the dock
;;   - transporter2  : Mobile transporter, starts at the dock
;;
;; Warehouse layout (connected graph):
;;
;;   shelf_A ---- shelf_B ---- shelf_C
;;      |            |            |
;;   picking_station |         staging_area
;;      |            |            |
;;    dock -------- aisle ------- dock
;;                (central)
;;
;; All connections are bidirectional.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem patient-order-001)
  (:domain pharmacy-warehouse)

  (:objects
    ;; Robots
    picker1       - robot
    transporter1  - robot
    transporter2  - robot

    ;; Locations
    dock              - location
    picking_station   - location
    aisle             - location
    staging_area      - location
    shelf_A           - location
    shelf_B           - location
    shelf_C           - location

    ;; Medication items
    insulin      - item
    amoxicillin  - item
    ibuprofen    - item
  )

  (:init
    ;; --- Robot starting positions ---
    (at picker1      picking_station)
    (at transporter1 dock)
    (at transporter2 dock)

    ;; --- Gripper states ---
    (gripper-empty picker1)
    (gripper-empty transporter1)
    (gripper-empty transporter2)

    ;; --- Item locations (bottles on shelves) ---
    (bottle-at insulin     shelf_A)
    (bottle-at amoxicillin shelf_B)
    (bottle-at ibuprofen   shelf_C)

    ;; --- Delivery dock ---
    (is-delivery-dock dock)

    ;; --- Warehouse connectivity (bidirectional) ---
    ;; Shelf row connections
    (connected shelf_A shelf_B)
    (connected shelf_B shelf_A)
    (connected shelf_B shelf_C)
    (connected shelf_C shelf_B)

    ;; Shelf to lower areas
    (connected shelf_A picking_station)
    (connected picking_station shelf_A)
    (connected shelf_B aisle)
    (connected aisle shelf_B)
    (connected shelf_C staging_area)
    (connected staging_area shelf_C)

    ;; Lower row connections
    (connected picking_station dock)
    (connected dock picking_station)
    (connected dock aisle)
    (connected aisle dock)
    (connected aisle staging_area)
    (connected staging_area aisle)
  )

  (:goal (and
    (order-fulfilled insulin)
    (order-fulfilled amoxicillin)
    (order-fulfilled ibuprofen)
  ))
)
