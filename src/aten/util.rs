use crate::core::*;

#[inline]
pub fn get_generator_or_default<'a, 'b: 'a>(
    gen: Option<&'a mut Generator>,
    default_gen: &'b mut Generator,
) -> &'a mut dyn GeneratorImpl {
    // match gen.as_mut() {
    //     Some(g) => check_generator(g),
    //     None => check_generator(default_gen),
    // }

    if let Some(g) = gen {
        check_generator(g)
    } else {
        check_generator(default_gen)
    }
}

#[inline(always)]
pub fn prod_intlist(array: &[usize]) -> usize {
    array.iter().product()
}
