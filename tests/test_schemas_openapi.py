"""
Test :mod:`porter.schemas.openapi`.

The tests in this suite rely to some extent on the fact that the
implementation can only even attempt validation after producing a valid spec
for use with fastjsonschema.  Thus, if validate() works as expected, then
to_openapi() must be working as well.

"""

import unittest

from porter.schemas import (String, Number, Integer, Boolean,
                            Array, Object,
                            RequestSchema, ResponseSchema)


class TestString(unittest.TestCase):
    def test_string(self):
        s = String('simple string')
        oi = s.to_openapi()[0]
        self.assertEqual(oi['type'], 'string')
        self.assertEqual(oi['description'], 'simple string')
        s.validate('test')
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data must be string'):
            s.validate(2)

        # check minLength
        s = String('long string', additional_params=dict(minLength=10))
        s.validate('1234567890')
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data must be longer'):
            s.validate('123456789')
        # check format
        s = String('date string', additional_params=dict(format='date'))
        s.validate('2001-01-01')
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data must be date'):
            s.validate('1/1/1')
        # check enum
        s = String('an enum', additional_params=dict(enum=['a', 'bc', 'def']))
        s.validate('def')
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data must be one of'):
            s.validate('ghij')

class TestNumber(unittest.TestCase):
    def test_number(self):
        # check type
        n = Number('a number')
        oi = n.to_openapi()[0]
        self.assertEqual(oi['type'], 'number')
        self.assertEqual(oi['description'], 'a number')
        n.validate(1)
        n.validate(2.0)
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data must be number'):
            n.validate('wrong')

        # check minimum
        n = Number('big number', additional_params=dict(minimum=1000))
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data must be bigger'):
            n.validate(999)
        # TODO: float/double distinction not supported by fastjsonschema?
        # https://github.com/CadentTech/porter/issues/30

class TestInteger(unittest.TestCase):
    def test_integer(self):
        # check type
        i = Integer('an integer')
        oi = i.to_openapi()[0]
        self.assertEqual(oi['type'], 'integer')
        self.assertEqual(oi['description'], 'an integer')
        i.validate(1)
        # TODO: xyz.0 validates as integer, desired behavior?
        # https://github.com/CadentTech/porter/issues/30
        i.validate(1.0)
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data must be integer'):
            i.validate(2.5)

        # check maximum
        i = Integer('small integer', additional_params=dict(maximum=3))
        i.validate(3)
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data must be smaller'):
            i.validate(4)

class TestBoolean(unittest.TestCase):
    def test_boolean(self):
        # check only True/False accepted
        b = Boolean('a boolean')
        oi = b.to_openapi()[0]
        self.assertEqual(oi['type'], 'boolean')
        self.assertEqual(oi['description'], 'a boolean')
        b.validate(False)
        b.validate(True)
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data must be boolean'):
            b.validate('true')

class TestArray(unittest.TestCase):
    def test_array(self):
        # check minItems
        a = Array('some ints', item_type=Integer(), additional_params=dict(minItems=3))
        a.validate([1,2,3])
        with self.assertRaisesRegex(
                ValueError, r'Schema validation failed: data must contain'):
            a.validate([1,2])

        # check nested constraint
        a = Array('some ints',
                  item_type=Integer(additional_params=dict(minimum=0)))
        a.validate([1, 0, 2])
        with self.assertRaisesRegex(
                ValueError, r'Schema validation failed: data\[1\] must be bigger'):
            a.validate([1, -1, 2])

class TestObject(unittest.TestCase):

    def setUp(self):
        # complex object
        self.o = Object(
            properties=dict(
                a=Object(
                    properties=dict(
                        aa=String(additional_params=dict(minLength=3)),
                        bb=Object(
                            properties=dict(
                                ccc=Array(
                                    item_type=Integer(),
                                    additional_params=dict(minItems=3))
                            )
                        )
                    )
                ),
                b=Object(additional_properties_type=Integer())
            ),
            additional_properties_type=Integer(),
        )

    def test_simple(self):
        # check flat object validation
        o = Object(
            properties=dict(
                a=String(),
                b=Number(),
                c=Integer(),
                d=Boolean()
            )
        )
        o.validate(dict(a='a', b=1.5, c=1, d=True))
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data.a must be string'):
            o.validate(dict(a=0, b=1.5, c=1, d=True))
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data.b must be number'):
            o.validate(dict(a='a', b='b', c=1, d=True))
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data.c must be integer'):
            o.validate(dict(a='a', b=1.5, c=1.5, d=True))
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data.d must be boolean'):
            o.validate(dict(a='a', b=1.5, c=1, d='true'))
        # check that empty object is invalid
        with self.assertRaisesRegex(
                ValueError, 'at least one of properties and additional_properties_type'):
            Object()


    def test_object_success(self):
        # check complex object that is valid
        self.o.validate(dict(
            a=dict(
                aa='123',
                bb=dict(
                    ccc=[1,2,3]
                )
            ),
            b=dict(x=1, y=2, z=3),
            c=3 * 10**8,
        ))

    def test_object_missing_key(self):
        # check missing nested field
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data.a must contain'):
            self.o.validate(dict(
                a=dict(
                    bb=dict(
                        ccc=[1,2]
                    )
                ),
                b=dict()
            ))

    def test_object_array_too_short(self):
        # check nested array too short
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data.a.bb.ccc must contain'):
            self.o.validate(dict(
                a=dict(
                    aa='123',
                    bb=dict(
                        ccc=[1,2]
                    )
                ),
                b=dict()
            ))

    def test_object_string_too_short(self):
        # check nested string too short
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data.a.aa must be longer'):
            self.o.validate(dict(
                a=dict(
                    aa='12',
                    bb=dict(
                        ccc=[1,2]
                    )
                ),
                b=dict()
            ))

    def test_object_bad_additional_property(self):
        # check additional property of wrong type
        with self.assertRaisesRegex(
                ValueError, 'Schema validation failed: data.pi must be integer'):
            self.o.validate(dict(
                a=dict(
                    aa='123',
                    bb=dict(
                        ccc=[1,2,3]
                    )
                ),
                b=dict(x=1, y=2, z=3),
                pi=3.1416,
            ))


class TestRequestSchema(unittest.TestCase):
    def test_request_body(self):
        # check that obj's schema is properly located within request body
        obj = Object(properties=dict(a=Number()))
        rb = RequestSchema(obj)
        schema = rb.to_openapi()[0]['requestBody']['content']['application/json']['schema']
        self.assertEqual(schema, obj.to_openapi()[0])

class TestResponseBody(unittest.TestCase):
    def test_response_body(self):
        # check that obj's schema is properly located within response body
        obj = Object(properties=dict(a=Number()))
        rb = ResponseSchema(status_code=200, api_obj=obj)
        schema = rb.to_openapi()[0][200]['content']['application/json']['schema']
        self.assertEqual(schema, obj.to_openapi()[0])


if __name__ == '__main__':
    unittest.main()
