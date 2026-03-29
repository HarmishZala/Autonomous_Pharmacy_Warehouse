;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; COMP0224 Coursework 2 — Task B: DEFECTIVE Domain (Critical Thinking Demo)
;; Student ID: 23013383
;;
;; PURPOSE: This file deliberately omits the (holding ?r ?i) effect from the
;; grasp action to demonstrate the Critical Thinking requirement:
;;
;;   "What happens to your plan if a Grasp action effect is missing?"
;;
;; ANALYSIS:
;;   The grasp action below is missing its primary ADD effect: (holding ?r ?i).
;;   This creates a logical deadlock:
;;
;;     1. grasp fires, consuming (bottle-at ?i ?l) and (gripper-empty ?r),
;;        but NEVER asserts (holding ?r ?i).
;;     2. The deliver action requires (holding ?r ?i) as a precondition.
;;     3. Since (holding ?r ?i) is never true, deliver can never be applied.
;;     4. The goal (order-fulfilled ?i) therefore becomes UNREACHABLE.
;;     5. Any STRIPS planner (FastForward, VHPOP) will report: NO PLAN FOUND.
;;
;;   Secondary consequence: the gripper state becomes permanently inconsistent.
;;   (gripper-empty ?r) is deleted, so the robot can never pick up another
;;   item either. The robot is stuck in a "ghost grasp" state — it performed
;;   the grasping motion but is holding nothing and cannot grasp again.
;;
;; This illustrates why the ADD list of an action is as critical as the
;; DELETE list: omitting an ADD effect can make goals unreachable even when
;; all preconditions are satisfiable.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain pharmacy-warehouse-defective)
  (:requirements :strips :typing)
  (:types robot location item)

  (:predicates
    (at ?r - robot ?l - location)
    (holding ?r - robot ?i - item)
    (gripper-empty ?r - robot)
    (bottle-at ?i - item ?l - location)
    (connected ?l1 - location ?l2 - location)
    (is-delivery-dock ?l - location)
    (order-fulfilled ?i - item)
  )

  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and
      (at ?r ?from)
      (connected ?from ?to)
    )
    :effect (and
      (at ?r ?to)
      (not (at ?r ?from))
    )
  )

  ;; -----------------------------------------------------------------------
  ;; DEFECTIVE grasp: (holding ?r ?i) ADD effect is intentionally MISSING.
  ;; The bottle is consumed from the shelf and the gripper is marked occupied,
  ;; but the robot is never recorded as holding the item.
  ;; Result: deliver's precondition (holding ?r ?i) can never be satisfied.
  ;; -----------------------------------------------------------------------
  (:action grasp
    :parameters (?r - robot ?i - item ?l - location)
    :precondition (and
      (at ?r ?l)
      (bottle-at ?i ?l)
      (gripper-empty ?r)
    )
    :effect (and
      ;; (holding ?r ?i)  <-- MISSING: this is the defect
      (not (bottle-at ?i ?l))
      (not (gripper-empty ?r))
    )
  )

  (:action deliver
    :parameters (?r - robot ?i - item ?l - location)
    :precondition (and
      (at ?r ?l)
      (holding ?r ?i)          ; requires holding — but grasp never sets this
      (is-delivery-dock ?l)
    )
    :effect (and
      (order-fulfilled ?i)
      (not (holding ?r ?i))
      (gripper-empty ?r)
    )
  )
)
