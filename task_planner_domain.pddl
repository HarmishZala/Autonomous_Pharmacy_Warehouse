;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; COMP0224 Coursework 2 — Task B: Pharmacy Warehouse Domain
;; Student ID: 23013383
;;
;; STRIPS domain for the autonomous pharmacy warehouse.
;; Three heterogeneous robots (1 Picker arm, 2 Transporters) must fulfil
;; medication orders by finding bottles on shelves, grasping them, and
;; delivering them to a dock.
;;
;; Key design decisions:
;;   - 'connected' predicate enforces valid navigation topology and prevents
;;     teleportation between arbitrary locations.
;;   - 'gripper-empty' predicate ensures a robot holds at most one item,
;;     reflecting the single-gripper physical constraint.
;;   - The 'move' action's delete list removes (at ?r ?from) to prevent the
;;     "bilocating robot" bug where a robot appears at two locations.
;;   - Terminal states: (order-fulfilled ?i) marks successful delivery;
;;     the gripper is freed so the robot can fulfil further orders.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain pharmacy-warehouse)
  (:requirements :strips :typing)
  (:types robot location item)

  (:predicates
    ;; Robot state
    (at ?r - robot ?l - location)          ; robot ?r is at location ?l
    (holding ?r - robot ?i - item)         ; robot ?r is holding item ?i
    (gripper-empty ?r - robot)             ; robot ?r has a free gripper

    ;; Item state
    (bottle-at ?i - item ?l - location)    ; item ?i is at location ?l

    ;; Environment topology
    (connected ?l1 - location ?l2 - location) ; ?l1 and ?l2 are adjacent

    ;; Location properties
    (is-delivery-dock ?l - location)       ; ?l is a valid delivery dock

    ;; Goal tracking
    (order-fulfilled ?i - item)            ; item ?i has been delivered
  )

  ;; -----------------------------------------------------------------------
  ;; ACTION: move
  ;; Robot navigates from one connected location to another.
  ;; The DELETE LIST removes (at ?r ?from) so the robot does not appear in
  ;; two places simultaneously (avoids the "bilocating robot" bug).
  ;; -----------------------------------------------------------------------
  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and
      (at ?r ?from)
      (connected ?from ?to)
    )
    :effect (and
      (at ?r ?to)              ; ADD: robot is now at destination
      (not (at ?r ?from))      ; DELETE: robot is no longer at origin
    )
  )

  ;; -----------------------------------------------------------------------
  ;; ACTION: grasp
  ;; Robot picks up a medicine bottle from its current location.
  ;; Preconditions enforce that (1) the robot is co-located with the bottle,
  ;; and (2) the gripper is free (single-item constraint).
  ;; The DELETE LIST removes (bottle-at ?i ?l) so the bottle does not remain
  ;; on the shelf after being picked up, and removes (gripper-empty ?r) so
  ;; the robot cannot pick up a second item.
  ;; -----------------------------------------------------------------------
  (:action grasp
    :parameters (?r - robot ?i - item ?l - location)
    :precondition (and
      (at ?r ?l)
      (bottle-at ?i ?l)
      (gripper-empty ?r)
    )
    :effect (and
      (holding ?r ?i)          ; ADD: robot now holds the item
      (not (bottle-at ?i ?l))  ; DELETE: bottle is no longer on the shelf
      (not (gripper-empty ?r)) ; DELETE: gripper is now occupied
    )
  )

  ;; -----------------------------------------------------------------------
  ;; ACTION: deliver
  ;; Robot places a held item at a delivery dock, fulfilling that part of
  ;; the patient order.
  ;; Preconditions enforce that the robot is at a valid dock and is holding
  ;; the item.
  ;; The DELETE LIST removes (holding ?r ?i) and restores (gripper-empty ?r)
  ;; so the robot can continue fulfilling additional orders.
  ;; -----------------------------------------------------------------------
  (:action deliver
    :parameters (?r - robot ?i - item ?l - location)
    :precondition (and
      (at ?r ?l)
      (holding ?r ?i)
      (is-delivery-dock ?l)
    )
    :effect (and
      (order-fulfilled ?i)     ; ADD: this item's order is fulfilled
      (not (holding ?r ?i))    ; DELETE: robot no longer holds the item
      (gripper-empty ?r)       ; ADD: gripper is free for next task
    )
  )
)
