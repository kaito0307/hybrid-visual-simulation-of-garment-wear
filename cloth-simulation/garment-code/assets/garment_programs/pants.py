from copy import deepcopy
import numpy as np

import pygarment as pyg
from assets.garment_programs.base_classes import BaseBottoms
from assets.garment_programs import bands


class PantPanel(pyg.Panel):
    def __init__(
            self, name, body, design, 
            length,
            waist, 
            hips,
            hips_depth,
            crotch_width,
            dart_position,
            match_top_int_to=None,
            hipline_ext=1,
            no_knee_expansion=False,
            double_dart=False) -> None:
        """
            Basic pant panel with option to be fitted (with darts)
        """
        super().__init__(name)

        flare = body['leg_circ'] * (design['flare']['v']  - 1) / 4 
        hips_depth = hips_depth * hipline_ext

        hip_side_incl = np.deg2rad(body['_hip_inclination'])
        dart_depth = hips_depth * 0.8 

        # Crotch cotrols
        crotch_depth_diff =  body['crotch_hip_diff']
        crotch_extention = crotch_width

        # eval pants shape

        # amount of extra fabric at waist
        w_diff = hips - waist   # Assume its positive since waist is smaller then hips
        # We distribute w_diff among the side angle and a dart 
        hw_shift = np.tan(hip_side_incl) * hips_depth
        # Small difference
        if hw_shift > w_diff:
            hw_shift = w_diff

        # --- Edges definition ---
        # Calculate knee region
        knee_expansion = design['knee_expansion']['v'] if 'knee_expansion' in design else 0
        knee_position = design['knee_position']['v'] if 'knee_position' in design else 0.5
        knee_length = design['knee_length']['v'] if 'knee_length' in design else 0
        knee_y = length * knee_position
        knee_start = knee_y - knee_length / 2
        knee_end = knee_y + knee_length / 2

        knee_dart_inside_width = design['knee_dart_inside_width']['v'] if 'knee_dart_inside_width' in design else 0
        knee_dart_inside_depth = design['knee_dart_inside_depth']['v'] if 'knee_dart_inside_depth' in design else 0
        knee_dart_inside_position = design['knee_dart_inside_position']['v'] if 'knee_dart_inside_position' in design else 0

        knee_dart_outside_width = design['knee_dart_outside_width']['v'] if 'knee_dart_outside_width' in design else 0
        knee_dart_outside_depth = design['knee_dart_outside_depth']['v'] if 'knee_dart_outside_depth' in design else 0
        knee_dart_outside_position = design['knee_dart_outside_position']['v'] if 'knee_dart_outside_position' in design else 0

        assert not((knee_dart_inside_depth > 0  or knee_dart_outside_depth > 0) and  knee_expansion > 0), 'Cannot have both knee dart and expansion at the same time!'

        if no_knee_expansion:
            knee_expansion = 0

        knee_y = length * knee_position
        # Right edge: bulge only in knee region
        if knee_expansion == 0:
            right_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
                [-flare, 0],
                [0, length],
                target_tan0=np.array([0, 1]),
                target_tan1=np.array([1, 0]),
                initial_guess=[0.5, 0]
            )
            right_bottom_int = right_bottom
            # Add knee dart to straight edge if requested
            if knee_dart_outside_depth > 0:
                dart_shape = pyg.EdgeSeqFactory.dart_shape(knee_dart_outside_width, knee_dart_outside_depth)
                edge_length = right_bottom.length()
                dart_offset = (knee_dart_outside_position if knee_dart_outside_position is not None else 0.5) * edge_length
                new_edges, _ = self.add_dart(dart_shape, right_bottom, offset=dart_offset, right=True)
                right_bottom = new_edges
                right_bottom_int = pyg.EdgeSequence(right_bottom[0], right_bottom[3])  # Don't include the dart in the interface
        else:
            x_base = -flare
            # Top to knee_start: straight
            pt_top = [x_base, 0]
            pt_knee_start = [x_base, knee_start]
            # Bulge curve: knee_start to knee_end
            pt_knee_bulge_start = [x_base, knee_start]
            pt_knee_bulge_mid = [x_base - knee_expansion / 2, knee_y]
            pt_knee_bulge_end = [x_base, knee_end]
            # Bottom: straight
            pt_bottom = [0, length]
            edge_top = pyg.Edge(pt_top, pt_knee_start)
            edge_bulge = pyg.CurveEdgeFactory.curve_3_points(
                pt_knee_bulge_start, pt_knee_bulge_end, pt_knee_bulge_mid
            )
            edge_bottom = pyg.Edge(pt_knee_bulge_end, pt_bottom)
            right_bottom = pyg.EdgeSequence(edge_top, edge_bulge, edge_bottom)
            right_bottom_int = right_bottom
        # Construct right_top using the correct end point
        if hasattr(right_bottom, 'end'):
            right_top_start = right_bottom.end
        else:
            right_top_start = right_bottom[-1].end
        right_top = pyg.CurveEdgeFactory.curve_from_tangents(
            right_top_start,
            [hw_shift, length + hips_depth],
            target_tan0=np.array([0, 1]),
            initial_guess=[0.5, 0]
        )
       
        top = pyg.Edge(
            right_top.end, 
            [w_diff + waist, length + hips_depth] 
        )

        crotch_top = pyg.Edge(
            top.end, 
            [hips, length + 0.45 * hips_depth]  # A bit higher than hip line
            # NOTE: The point should be lower than the minimum rise value (0.5)
        )
        crotch_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
            crotch_top.end,
            [hips + crotch_extention, length - crotch_depth_diff], 
            target_tan0=np.array([0, -1]),
            target_tan1=np.array([1, 0]),
            initial_guess=[0.5, -0.5] 
        )

        # Left edge: bulge only in knee region
        if knee_expansion == 0:
            left = pyg.CurveEdgeFactory.curve_from_tangents(
                crotch_bottom.end,
                [crotch_bottom.end[0] - 2 + flare, min(0, length - crotch_depth_diff * 1.5)],
                target_tan0=np.array([1, 0]),
                target_tan1=np.array([0, -1]),
                initial_guess=[0.5, 0]
            )
            left_int = left
            # Add knee dart to straight edge if requested
            if knee_dart_inside_depth > 0:
                dart_shape = pyg.EdgeSeqFactory.dart_shape(knee_dart_inside_width, knee_dart_inside_depth)
                edge_length = left.length()
                dart_offset = (knee_dart_inside_position if knee_dart_inside_position is not None else 0.5) * edge_length
                new_edges, _ = self.add_dart(dart_shape, left, offset=dart_offset, right=True)
                left = new_edges
                left_int = pyg.EdgeSequence([left[0], left[3]])  # No interface for left edge
            # todo: remove the darts from interface.
        else:
            x_base = crotch_bottom.end[0] - 2 + flare
            # Top to knee_start: straight
            pt_top = crotch_bottom.end
            pt_knee_start = [pt_top[0], knee_end]
            # Bulge curve: knee_start to knee_end
            pt_knee_bulge_start = [pt_top[0], knee_end]
            pt_knee_bulge_mid = [pt_top[0] + knee_expansion / 2, knee_y]
            pt_knee_bulge_end = [pt_top[0], knee_start]
            # Bottom: straight
            pt_bottom = [x_base, min(0, length - crotch_depth_diff * 1.5)]
            edge_top = pyg.Edge(pt_top, pt_knee_start)
            edge_bulge = pyg.CurveEdgeFactory.curve_3_points(
                pt_knee_bulge_start, pt_knee_bulge_end, pt_knee_bulge_mid
            )
            edge_bottom = pyg.Edge(pt_knee_bulge_end, pt_bottom)
            left = pyg.EdgeSequence(edge_top, edge_bulge, edge_bottom)
            left_int = left

        # Update interfaces to match new edge sequences after dart insertion
        self.edges = pyg.EdgeSequence(
            right_bottom, right_top, top, crotch_top, crotch_bottom, left
        ).close_loop()
        bottom = self.edges[-1]

        # Default placement
        self.set_pivot(crotch_bottom.end)
        self.translation = [-0.5, - hips_depth - crotch_depth_diff + 5, 0] 

        # Out interfaces (easier to define before adding a dart)
        outside_edges = pyg.EdgeSequence(right_bottom, right_top)
        outside_int = pyg.EdgeSequence(right_bottom_int, right_top)
        n_outside = len(outside_int)
        ruffle_outside = [1] * n_outside
        # Optionally, set the last ruffle value to hipline_ext if you want to preserve the original logic
        if n_outside > 1:
            ruffle_outside[-1] = hipline_ext
        self.interfaces = {
            'outside': pyg.Interface(
                self, 
                outside_int,
                ruffle=ruffle_outside),
            'crotch': pyg.Interface(self, pyg.EdgeSequence(crotch_top, crotch_bottom)),
            'inside': pyg.Interface(self, left_int),
            'bottom': pyg.Interface(self, bottom)
        }

        # Add top dart
        # NOTE: Ruffle indicator to match to waistline proportion for correct balance line matching
        dart_width = w_diff - hw_shift  
        if w_diff > hw_shift:
            top_edges, int_edges = self.add_darts(
                top, dart_width, dart_depth, dart_position, double_dart=double_dart)
            self.interfaces['top'] = pyg.Interface(
                self, int_edges, 
                ruffle=waist / match_top_int_to if match_top_int_to is not None else 1.
            ) 
            self.edges.substitute(top, top_edges)
        else:
            self.interfaces['top'] = pyg.Interface(
                self, top, 
                ruffle=waist / match_top_int_to if match_top_int_to is not None else 1.
        ) 
        

    def add_darts(self, top, dart_width, dart_depth, dart_position, double_dart=False):
        
        if double_dart:
            # TODOLOW Avoid hardcoding for matching with the top?
            dist = dart_position * 0.5  # Dist between darts -> dist between centers
            offsets_mid = [
                - (dart_position + dist / 2 + dart_width / 2 + dart_width / 4),   
                - (dart_position - dist / 2) - dart_width / 4,
            ]

            darts = [
                pyg.EdgeSeqFactory.dart_shape(dart_width / 2, dart_depth * 0.9), # smaller
                pyg.EdgeSeqFactory.dart_shape(dart_width / 2, dart_depth)
            ]
        else:
            offsets_mid = [
                - dart_position - dart_width / 2,
            ]
            darts = [
                pyg.EdgeSeqFactory.dart_shape(dart_width, dart_depth)
            ]
        top_edges, int_edges = pyg.EdgeSequence(top), pyg.EdgeSequence(top)

        for off, dart in zip(offsets_mid, darts):
            left_edge_len = top_edges[-1].length()
            top_edges, int_edges = self.add_dart(
                dart,
                top_edges[-1],
                offset=left_edge_len + off,
                edge_seq=top_edges, 
                int_edge_seq=int_edges
            )

        return top_edges, int_edges
        

class PantsHalf(BaseBottoms):
    def __init__(self, tag, body, design, rise=None) -> None:
        super().__init__(body, design, tag, rise=rise)
        design = design['pants']
        self.rise = design['rise']['v'] if rise is None else rise
        waist, hips_depth, waist_back = self.eval_rise(self.rise)

        # NOTE: min value = full sum > leg curcumference
        # Max: pant leg falls flat from the back
        # Mostly from the back side
        # => This controls the foundation width of the pant
        min_ext = body['leg_circ'] - body['hips'] / 2  + 5  # 2 inch ease: from pattern making book 
        front_hip = (body['hips'] - body['hip_back_width']) / 2
        crotch_extention = min_ext * design['width']['v']  
        front_extention = front_hip / 4    # From pattern making book
        back_extention = crotch_extention - front_extention

        length, cuff_len = design['length']['v'], design['cuff']['cuff_len']['v']
        if design['cuff']['type']['v']: 
            if length - cuff_len < design['length']['range'][0]:   # Min length from paramss
                # Cannot be longer then a pant
                cuff_len = length - design['length']['range'][0]
            # Include the cuff into the overall length, 
            # unless the requested length is too short to fit the cuff 
            # (to avoid negative length)
            length -= cuff_len
        length *= body['_leg_length']
        cuff_len *= body['_leg_length']

        self.front = PantPanel(
            f'pant_f_{tag}', body, design,
            length=length,
            waist=(waist - waist_back) / 2,
            hips=(body['hips'] - body['hip_back_width']) / 2,
            hips_depth=hips_depth,
            dart_position = body['bust_points'] / 2,
            crotch_width=front_extention,
            match_top_int_to=(body['waist'] - body['waist_back_width']) / 2
            ).translate_by([0, body['_waist_level'] - 5, 25])
        self.back = PantPanel(
            f'pant_b_{tag}', body, design,
            length=length,
            waist=waist_back / 2,
            hips=body['hip_back_width'] / 2,
            hips_depth=hips_depth,
            hipline_ext=1.1,
            dart_position = body['bum_points'] / 2,
            crotch_width=back_extention,
            match_top_int_to=body['waist_back_width'] / 2,
            no_knee_expansion=True,
            double_dart=True
            ).translate_by([0, body['_waist_level'] - 5, -20])

        self.stitching_rules = pyg.Stitches(
            (self.front.interfaces['outside'], self.back.interfaces['outside']),
            (self.front.interfaces['inside'], self.back.interfaces['inside'])
        )

        # add a cuff
        # TODOLOW This process is the same for sleeves -- make a function?
        if design['cuff']['type']['v']:
            
            pant_bottom = pyg.Interface.from_multiple(
                self.front.interfaces['bottom'],
                self.back.interfaces['bottom'])

            # Copy to avoid editing original design dict
            cdesign = deepcopy(design)
            cdesign['cuff']['b_width'] = {}
            cdesign['cuff']['b_width']['v'] = pant_bottom.edges.length() / design['cuff']['top_ruffle']['v']
            cdesign['cuff']['cuff_len']['v'] = cuff_len

            # Init
            cuff_class = getattr(bands, cdesign['cuff']['type']['v'])
            self.cuff = cuff_class(f'pant_{tag}', cdesign)

            # Position
            self.cuff.place_by_interface(
                self.cuff.interfaces['top'],
                pant_bottom,
                gap=5,
                alignment='left'
            )

            # Stitch
            self.stitching_rules.append((
                pant_bottom,
                self.cuff.interfaces['top'])
            )

        self.interfaces = {
            'crotch_f': self.front.interfaces['crotch'],
            'crotch_b': self.back.interfaces['crotch'],
            'top_f': self.front.interfaces['top'], 
            'top_b': self.back.interfaces['top'] 
        }

    def length(self):
        if self.design['pants']['cuff']['type']['v']:
            return self.front.length() + self.cuff.length()
        
        return self.front.length()

class Pants(BaseBottoms):
    def __init__(self, body, design, rise=None) -> None:
        super().__init__(body, design)

        self.right = PantsHalf('r', body, design, rise)
        self.left = PantsHalf('l', body, design, rise).mirror()

        self.stitching_rules = pyg.Stitches(
            (self.right.interfaces['crotch_f'], self.left.interfaces['crotch_f']),
            (self.right.interfaces['crotch_b'], self.left.interfaces['crotch_b']),
        )

        self.interfaces = {
            'top_f': pyg.Interface.from_multiple(
                self.right.interfaces['top_f'], self.left.interfaces['top_f']),
            'top_b': pyg.Interface.from_multiple(
                self.right.interfaces['top_b'], self.left.interfaces['top_b']),
            # Some are reversed for correct connection
            'top': pyg.Interface.from_multiple(   # around the body starting from front right
                self.right.interfaces['top_f'].flip_edges(),
                self.left.interfaces['top_f'].reverse(with_edge_dir_reverse=True),
                self.left.interfaces['top_b'].flip_edges(),
                self.right.interfaces['top_b'].reverse(with_edge_dir_reverse=True), # Flips the edges and restores the direction
            )
        }

    def get_rise(self):
        return self.right.get_rise()
    
    def length(self):
        return self.right.length()
