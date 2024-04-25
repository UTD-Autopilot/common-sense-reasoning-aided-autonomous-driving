%DATA REPRESENTATION

%property(vehicle, Frame, Object_id, Action, VX, VY, R, C1X, C1Y, C2X, C2Y).
%property(traffic_light, Frame, Object_id, First_Vehicle_ID, Color, C1X, C1Y, C2X, C2Y).
%property(intersection, Frame, Object_id, C1X, C1Y, C2X, C2Y).
%property(obstacle, Frame, Object_id, C1X, C1Y, C2X, C2Y).
%vehicles(Frame, Vehicles).
%frames(Frames).
%change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y).
%driverLocation(T, CX, CY).
%ground_light(Frame, Value).
%pred_light(Frame, Prediction)
%driver_rotation(Frame, Rotation).


writefacts():-
	frames(Frames),
	writefacts(Frames).

writefacts([Frame|T]):-
	open('C:\\Users\\keega\\Downloads\\record_20240106_part1\\Town04_200npcs_1\\agents\\0\\outputC.txt',append,Out),
	write(Out, Frame), nl(Out),
	consistent_intersection_eval(Frame, R),
	ground_light(Frame, R1),
	pred_light(Frame, R2),
	red_logic_eval(Frame, R3),
	explainable_lane_ground_eval(Frame, R4),
	explainable_lane_change_eval(Frame, R5),
	write(Out, R), nl(Out), 
	write(Out, R1), nl(Out),
	write(Out, R2), nl(Out),
	write(Out, R3), nl(Out),
	write(Out, R4), nl(Out),
	write(Out, R5), nl(Out),
	close(Out),
	writefacts(T).

writefacts([Frame|T]):-
	!,
	writefacts(T).

writeFacts([]).

consistent_intersection_eval(Frame, true):-
	consistent_intersection(Frame).
consistent_intersection_eval(Frame, false).

explainable_lane_ground_eval(Frame, true).

explainable_lane_change_eval(Frame, true):-
	explainable_lane_change(Frame).
explainable_lane_change_eval(Frame, false).

red_ego_eval(Frame, true):-
	red_ego_traffic_light(Frame).
red_ego_eval(Frame, false):-
	green_ego_traffic_light(Frame).
red_ego_eval(Frame, none).


red_logic_eval(Frame, none):-
	\+property(intersection, Frame, _, _, _, _, _).
red_logic_eval(Frame, none):-
	\+change_action_cluster(Frame).
red_logic_eval(Frame, true):-
	red_traffic_light(Frame).
red_logic_eval(Frame, false):-
	green_traffic_light(Frame).


%ACTION SELECTION


%Following rule checks if the scenario at the given frame is consistent. (Valid scenario that contains no violations of rules and is explainable)
consistent(T):-
	explainable_lane_change(T),
	consistent_intersection(T).
consistent(T):-
	\+explainable_lane_change(T),
	print('Can\'t find explanation for lane change. Possible false negative for a nearby obstacle or intersection.').
consistent(T):-
	\+consistent_intersection(T),
	print('Violation of traffic light rules. Color of a traffic light has been mislabeled.').

%Explainable Scenarios for collective lane changes. An obstacle or intersection makes it explainable. If none are detected perhaps there is a false negative in the prediction.
explainable_lane_change(T):-
	change_action_cluster(FrameStart, FrameEnd, change_lane_left, _, _, _, _, _),
	T >= FrameStart,
	T =< FrameEnd,
	property(intersection, T, _, _, _, _, _).
explainable_lane_change(T):-
	change_action_cluster(FrameStart, FrameEnd, change_lane_right, _, _, _, _, _),
	T >= FrameStart,
	T =< FrameEnd,
	property(intersection, T, _, _, _, _, _).
explainable_lane_change(T):-
	change_action_cluster(FrameStart, FrameEnd, change_lane_left, _, _, _, _, _),
	T >= FrameStart,
	T =< FrameEnd,
	property(obstacle, T, _, _, _, _, _).
explainable_lane_change(T):-
	change_action_cluster(FrameStart, FrameEnd, change_lane_right, _, _, _, _, _),
	T >= FrameStart,
	T =< FrameEnd,
	property(obstacle, T, _, _, _, _, _).
explainable_lane_change(T):-
	\+ change_action_cluster(FrameStart, FrameEnd, change_lane_right, _, _, _, _, _),
	\+ change_action_cluster(FrameStart, FrameEnd, change_lane_right, _, _, _, _, _).

%Consistent Scenarios for traffic lights and intersections. The logic must reason a color based on collective behavior for each traffic light and the prediction of the deep learning model must match.
consistent_intersection(T):-
	ground_light(T, true),
	property(intersection, T, _, _, _, _, _),
	red_traffic_light(T).
consistent_intersection(T):-
	ground_light(T, false),
	property(intersection, T, _, _, _, _, _),
	\+ red_traffic_light(T).
consistent_intersection(T):-
	\+property(intersection, Frame, Object_id, C1X, C1Y, C2X, C2Y).

%Temp for testing
inconsistent_intersection(T):-
	red_ego_traffic_light(T),
	property(intersection, T, _, _, _, _, _),
	\+ red_traffic_light(T).

inconsistent_intersection(T):-
	green_ego_traffic_light(T),
	property(intersection, T, _, _, _, _, _),
	red_traffic_light(T).

red_ego_traffic_light(T):-
	driverLocation(T, C1X, C1Y),
	T2 is T-10,
	driverLocation(T2, C2X, C2Y),
	((abs(C1X) - abs(C2X)) + (abs(C1Y) - abs(C2Y))) < 1,
	property(intersection, T, _, _, _, _, _).

green_ego_traffic_light(T):-
	driverLocation(T, C1X, C1Y),
	T2 is T-10,
	driverLocation(T2, C2X, C2Y),
	((abs(C1X) - abs(C2X)) + (abs(C1Y) - abs(C2Y))) > 1,
	property(intersection, T, _, _, _, _, _).

green_traffic_light(T):-
	property(intersection, T, _, _, _, _, _), 
	\+ red_traffic_light(T).

red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	stopped_vehicle_in_front(T), 
	%print('Red light for ego vehicle. Vehicle directly in front is stopped at an intersection'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_up_straight(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles moving forward.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_up_right(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles moving forward.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_up_left(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles on the other side are turning left.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_down_straight(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles moving forward.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_down_left(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles on the other side are turning left.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_down_right(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles on the other side are turning left.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_left_straight(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles moving forward.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_left_left(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles on the other side are turning left.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_left_right(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles on the other side are turning left.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_right_straight(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles moving forward.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_right_left(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles on the other side are turning left.'),
	\+ab(T).
red_traffic_light(T):- 
	property(intersection, T, _, _, _, _, _), 
	collective_right_right(T), 
	\+ ego_collective(T),
	%print('Red light for ego vehicle. Group of vehicles on the other side are turning left.'),
	\+ab(T).


%SUPPORTING RELATIONS



stopped_vehicle_in_front(T):-
	driverLocation(T, X, Y),
	driver_rotation(T, R),
	R > -95,
	R < -85,
	property(vehicle, T, First_Vehicle_ID, _, VX, VY, R2, C1X, C1Y, C2X, C2Y),
	property(traffic_light, T, _, First_Vehicle_ID, _, _, _, _, _),
	abs(VX) < (1/10),
	abs(VY) < (1/10),
	abs((R - R2)) < 5,
	C1Y > Y,
	abs((C1Y - Y)) < 10,
	abs((C1Y - Y)) > (1/10).

stopped_vehicle_in_front(T):-
	driverLocation(T, X, Y),
	driver_rotation(T, R),
	R > 95,
	R < 85,
	property(vehicle, T, First_Vehicle_ID, _, VX, VY, R2, C1X, C1Y, C2X, C2Y),
	property(traffic_light, T, _, First_Vehicle_ID, _, _, _, _, _),
	abs(VX) < 1,
	abs(VY) < 1,
	abs((R - R2)) < 5,
	C1Y < Y,
	abs((C1Y - Y)) < 10,
	abs((C1Y - Y)) > (1/10).

stopped_vehicle_in_front(T):-
	driverLocation(T, X, Y),
	driver_rotation(T, R),
	R > -5,
	R < 5,
	property(vehicle, T, First_Vehicle_ID, _, VX, VY, R2, C1X, C1Y, C2X, C2Y),
	property(traffic_light, T, _, First_Vehicle_ID, _, _, _, _, _),
	abs(VX) < 1,
	abs(VY) < 1,
	abs((R - R2)) < 5,
	X > C1X,
	abs((C1X - X)) < 10,
	abs((C1X - X)) > (1/10).

stopped_vehicle_in_front(T):-
	driverLocation(T, X, Y),
	driver_rotation(T, R),
	(R > 175; R < -175),
	property(vehicle, T, First_Vehicle_ID, _, VX, VY, R2, C1X, C1Y, C2X, C2Y),
	property(traffic_light, T, _, First_Vehicle_ID, _, _, _, _, _),
	abs(VX) < 1,
	abs(VY) < 1,
	abs((R - R2)) < 5,
	C2X > X,
	abs((C2X - X)) < 10,
	abs((C2X - X)) > (1/10).


ego_collective(T):-
	change_action_cluster(FrameStart, FrameEnd, _, Ego, _, _, _, _),
	driverLocation(T, C1X, C1Y),
	T2 is T-10,
	driverLocation(T2, C2X, C2Y),
	(abs((abs(C1X) - abs(C2X))) + abs((abs(C1Y) - abs(C2Y)))) > 0,
	T >= FrameStart,
	T =< FrameEnd,
	Ego = true.

ego_collective(T):-
	change_action_cluster(FrameStart, FrameEnd, _, Ego, _, _, _, _),
	driverLocation(T, C1X, C1Y),
	T2 is T-10,
	\+driverLocation(T2, C2X, C2Y),
	T >= FrameStart,
	T =< FrameEnd,
	Ego = true.

collective_up_left(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = left,
	intersection_up_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_up_straight(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = straight,
	intersection_up_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_up_right(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = right,
	intersection_up_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_up_stop(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = stop,
	intersection_up_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_down_left(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = left,
	intersection_down_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_down_straight(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = straight,
	intersection_down_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_down_right(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = right,
	intersection_down_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_down_stop(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = stop,
	intersection_down_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_right_left(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = left,
	intersection_right_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_right_straight(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = straight,
	intersection_right_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_right_right(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = right,
	intersection_right_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_right_stop(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	T >= FrameStart,
	T =< FrameEnd,
	Action = stop,
	intersection_right_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_left_left(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = left,
	intersection_left_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_left_straight(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = straight,
	intersection_left_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_left_right(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = right,
	intersection_left_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

collective_left_stop(T):-
	property(intersection, T, Object_id, _, _, _, _), 
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	T >= FrameStart,
	T =< FrameEnd,
	Action = stop,
	intersection_left_cluster(T, Object_id, C1X, C1Y, C2X, C2Y).

intersection_up_cluster(Frame, Object_id, C1X, C1Y, C2X, C2Y):-
	property(intersection, Frame, Object_id, _, _, _, _ ), 
	property(intersection_side_up, Frame, Object_id, IS1X, IS1Y, W, H),
	IS1Y+H < C1Y.

intersection_down_cluster(Frame, Object_id, C1X, C1Y, C2X, C2Y):-
	property(intersection, Frame, Object_id, _, _, _, _ ), 
	property(intersection_side_down, Frame, Object_id, IS1X, IS1Y, W, H),
	IS1Y > C2Y.

intersection_right_cluster(Frame, Object_id, C1X, C1Y, C2X, C2Y):-
	property(intersection, Frame, Object_id, _, _, _, _ ), 
	property(intersection_side_right, Frame, Object_id, IS1X, IS1Y, W, H),
	IS1X > C2X.

intersection_left_cluster(Frame, Object_id, C1X, C1Y, C2X, C2Y):-
	property(intersection, Frame, Object_id, _, _, _, _ ), 
	property(intersection_side_left, Frame, Object_id, IS1X, IS1Y, W, H),
	IS1X+W < C1X.

change_action_cluster(T):-
	change_action_cluster(FrameStart, FrameEnd, Action, Ego, C1X, C1Y, C2X, C2Y),
	Ego = false,
	Action \= changelaneright,
	Action \= changelaneleft,
	T >= FrameStart,
	T =< FrameEnd.

ab(T) :- false.



